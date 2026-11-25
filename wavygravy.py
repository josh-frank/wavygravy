#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# wavygravy.py — hide a short UTF-8 message in an audio file using self-locating
# "ping" beacons and a repeating carousel of coded chunks. (v0 / lean draft)
#
#   KOMETA_PASSWORD=... wavygravy.py encode  cover.wav "stop spying on me" out.wav
#   KOMETA_PASSWORD=... wavygravy.py decode  out.wav [message.txt]
#                       wavygravy.py capacity cover.wav
#
# Design, in one paragraph:
#   The timeline is cut into FRAMES. Each frame starts with a PING: a pair of
#   linear chirps (one up-sweep, one down-sweep) whose *order* is reversed on
#   the frame that opens each lap of the carousel. Pings are found by a global
#   matched-filter scan with no notion of "where the stream starts" — a
#   decoder can walk in anywhere. The up/down chirp timing asymmetry gives a
#   free, per-frame estimate of frequency offset and channel phase (the
#   classic symmetric-sweep trick), so there is no drift-tracking loop and no
#   frequency grid search: every frame re-synchronizes itself from scratch.
#   After each ping comes a DATA field: a fixed number of BPSK symbols spread
#   by a keyed noise carrier that depends on the *slot number*, not on
#   absolute position in the file. Slot 0 always carries the (tiny) header;
#   slots 1..N carry the FEC'd ciphertext, round-robin, lap after lap. A
#   decoder just counts frames since the last boundary ping to know which
#   slot it is looking at, and sums soft values across laps before running
#   Viterbi — so gaps self-heal on the next lap, and partial audio still
#   contributes.
#
# What got dropped vs. a from-scratch design, on purpose, for leanness:
#   - No global interleaver: carousel repetition across laps is the burst-
#     error defense now, instead of bit-interleaving within one pass.
#   - No Costas phase-tracking loop: each frame gets a fresh phase/frequency
#     fix from its own ping pair instead of tracking one continuously.
#   - No --simulate harness here; test channel degradation out-of-band.
#   - No --max-freq-offset knob: offset is measured per frame, not searched.
#
# Known trade-off: chirps are a more *recognizable* shape in a spectrogram
# than keyed noise would be (stealth-vs-robustness dial mentioned in the
# design notes) — acceptable for this draft, worth revisiting later.
#
# Ping/data strength budget: pings ride PING_BOOST_DB louder than the data
# field. Measured against broadband-noise covers, a ping at the *same*
# level as the data symbols is only marginally louder than the cover's own
# chance correlation with the chirp template — true pings and noise-only
# peaks land in the same z-score range and detection gets unreliable. Since
# losing a ping just costs one slot's worth of one lap (the carousel
# recovers it next time around) while losing *sync entirely* costs the
# whole message, it is worth spending more of the strength budget on the
# beacon than on the payload.
# ─────────────────────────────────────────────────────────────────────────────

import argparse, hashlib, hmac, os, sys, getpass, subprocess, tempfile
from math import gcd, ceil

import numpy as np
from scipy import signal
from scipy.io import wavfile
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

# ── constants ────────────────────────────────────────────────────────────────

WORK_FS       = 8000
DEFAULT_BAND  = (300.0, 2700.0)
KDF_N         = 2 ** 14

CHIRP_DUR     = 0.25                          # seconds, each leg of a ping
CHIRP_LEN     = int(CHIRP_DUR * WORK_FS)
CHIRP_F0      = 350.0                         # sweep range, inside DEFAULT_BAND
CHIRP_F1      = 2650.0
PING_Z        = 7.0                           # ping-detect threshold (std devs)
PING_BOOST_DB = 6.0                           # pings ride louder than data: see note below
LEG_SEARCH    = 40                            # samples, fine-align search window

REDUNDANCY    = {"low": 128, "medium": 256, "high": 512}   # samples per data bit
LEN_BYTES     = 4
SALT_BYTES    = 6
HDR_BYTES     = LEN_BYTES + SALT_BYTES
TAG_BYTES     = 16
ENV_WIN       = 400
ENV_FLOOR     = 0.1
DEFAULT_STRENGTH_DB = 18.0

HDR_CODED_BITS   = 2 * (8 * HDR_BYTES + 6)    # rate-1/2 conv code + 6 tail bits
SYMS_PER_FRAME   = HDR_CODED_BITS             # every slot (incl. header) is this long

class WavyError(Exception):
    pass

# ── keys & keyed pseudo-randomness ───────────────────────────────────────────

def carrier_key(password):
    return hashlib.scrypt(password.encode(), salt=b"wavygravy/carrier/v1",
                          n=KDF_N, r=8, p=1, dklen=32)

def subkey(ck, label):
    return hmac.new(ck, label, hashlib.sha256).digest()

def keystream(key32, nbytes):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    enc = Cipher(algorithms.ChaCha20(key32, bytes(16)), mode=None).encryptor()
    return enc.update(bytes(nbytes))

def keyed_bits(key32, n):
    ks = np.frombuffer(keystream(key32, (n + 7) // 8), dtype=np.uint8)
    return np.unpackbits(ks)[:n]

def msg_key(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=KDF_N, r=8, p=1, dklen=32)

# ── DSP helpers ──────────────────────────────────────────────────────────────

def bp_fir(band):
    return signal.firwin(401, band, pass_zero=False, fs=WORK_FS)

def bandpass(x, band):
    return signal.fftconvolve(x, bp_fir(band), mode="same")

def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    g = gcd(int(fs_in), int(fs_out))
    return signal.resample_poly(x, int(fs_out) // g, int(fs_in) // g)

def envelope(xb):
    m = np.sqrt(np.maximum(
        signal.fftconvolve(xb ** 2, np.ones(ENV_WIN) / ENV_WIN, mode="same"), 0))
    return np.maximum(m, ENV_FLOOR * (np.sqrt(np.mean(xb ** 2)) + 1e-12))

def slot_carrier(ck, slot, n, band):
    """Keyed band-limited noise carrier for one carousel slot, unit RMS.
    Keyed by slot number (not absolute sample position) so a decoder that
    joins mid-stream can despread a slot the moment it knows its number."""
    key = subkey(ck, b"pn:" + str(int(slot)).encode())
    chips = 1.0 - 2.0 * keyed_bits(key, n)
    p = bandpass(chips, band)
    return p / np.sqrt(np.sum(bp_fir(band) ** 2))

# ── audio I/O ────────────────────────────────────────────────────────────────

def _to_float(data):
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float64) / float(np.iinfo(data.dtype).max + 1)
    return data.astype(np.float64)

def read_audio(path):
    try:
        fs, data = wavfile.read(path)
    except Exception:
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, "in.wav")
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path,
                                "-f", "wav", "-acodec", "pcm_s16le", tmp],
                               check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError):
                raise WavyError(f"cannot read {path} (not a PCM WAV, and ffmpeg could not convert it)")
            fs, data = wavfile.read(tmp)
    return int(fs), _to_float(data)

def write_audio(path, fs, x):
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 0.999:
        x = x * (0.999 / peak)
    wavfile.write(path, int(fs), np.round(x * 32767.0).astype(np.int16))

def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x

# ── forward error correction: K=7, r=1/2 convolutional code, soft Viterbi ────
# (unchanged from the kometawav draft — this part already pulled its weight)

_G1, _G2 = 0o133, 0o171

def _parity(v):
    return bin(v).count("1") & 1

_NS     = np.arange(64)
_PRED   = np.stack([((_NS & 31) << 1) | 0, ((_NS & 31) << 1) | 1])
_INBIT  = _NS >> 5
_EXP    = np.zeros((2, 64, 2))
for _b in range(2):
    for _ns in range(64):
        _reg = (int(_INBIT[_ns]) << 6) | int(_PRED[_b, _ns])
        _EXP[_b, _ns] = (1 - 2 * _parity(_reg & _G1), 1 - 2 * _parity(_reg & _G2))

def conv_encode(bits):
    bits = np.concatenate([np.asarray(bits, dtype=np.uint8), np.zeros(6, np.uint8)])
    out, state = [], 0
    for u in bits:
        reg = (int(u) << 6) | state
        out += [_parity(reg & _G1), _parity(reg & _G2)]
        state = reg >> 1
    return np.array(out, dtype=np.uint8)

def viterbi_decode(soft):
    T = len(soft) // 2
    s1, s2 = soft[0::2][:T], soft[1::2][:T]
    pm = np.full(64, -1e18); pm[0] = 0.0
    dec = np.zeros((T, 64), dtype=bool)
    for t in range(T):
        m0 = pm[_PRED[0]] + s1[t] * _EXP[0, :, 0] + s2[t] * _EXP[0, :, 1]
        m1 = pm[_PRED[1]] + s1[t] * _EXP[1, :, 0] + s2[t] * _EXP[1, :, 1]
        dec[t] = m1 > m0
        pm = np.where(dec[t], m1, m0)
    bits = np.zeros(T, dtype=np.uint8)
    ns = 0
    for t in range(T - 1, -1, -1):
        bits[t] = _INBIT[ns]
        ns = _PRED[int(dec[t, ns]), ns]
    return bits[:-6]

def fec_encode(data_bytes):
    bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
    return conv_encode(bits)

def fec_decode(soft):
    bits = viterbi_decode(soft)
    return np.packbits(bits).tobytes()

# ── pings: a chirp pair whose leg order marks carousel-lap boundaries ───────

def _make_chirp(f0, f1, n):
    t = np.arange(n) / WORK_FS
    k = (f1 - f0) / (n / WORK_FS)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    return np.sin(phase)

_UP   = _make_chirp(CHIRP_F0, CHIRP_F1, CHIRP_LEN)
_DOWN = _make_chirp(CHIRP_F1, CHIRP_F0, CHIRP_LEN)
_UP_A, _DOWN_A = signal.hilbert(_UP), signal.hilbert(_DOWN)   # analytic refs for phase/freq fix

def ping_waveform(boundary):
    """boundary=True marks the frame that opens a carousel lap (slot 0)."""
    return np.concatenate([_DOWN, _UP]) if boundary else np.concatenate([_UP, _DOWN])

def find_pings(xn, z_thresh=PING_Z):
    """Global matched-filter scan, no assumption about where the stream
    starts. Returns [(start_sample, boundary_flag, z), ...] time-ordered."""
    normal = np.concatenate([_UP, _DOWN])
    boundary = np.concatenate([_DOWN, _UP])
    cn = signal.fftconvolve(xn, normal[::-1], mode="valid")
    cb = signal.fftconvolve(xn, boundary[::-1], mode="valid")
    score = np.maximum(np.abs(cn), np.abs(cb))
    z = (score - score.mean()) / (score.std() + 1e-12)
    min_gap = int(1.5 * CHIRP_LEN)
    idx = np.where(z >= z_thresh)[0]
    idx = idx[np.argsort(-z[idx])]
    taken = np.zeros(len(score), dtype=bool)
    pings = []
    for i in idx:
        lo, hi = max(0, i - min_gap), i + min_gap
        if taken[lo:hi].any():
            continue
        taken[lo:hi] = True
        is_boundary = abs(cb[i]) > abs(cn[i])
        pings.append((int(i), bool(is_boundary), float(z[i])))
    pings.sort(key=lambda p: p[0])
    return pings

def _leg_peak(xa, start_guess, ref_a, search=LEG_SEARCH):
    """Fine-align one chirp leg against its analytic reference; returns
    (sample offset, complex correlation value) at the best nearby lag."""
    best = None
    for d in range(-search, search + 1):
        s = start_guess + d
        if s < 0 or s + len(ref_a) > len(xa):
            continue
        z = np.vdot(ref_a, xa[s:s + len(ref_a)])
        if best is None or abs(z) > abs(best[1]):
            best = (s, z)
    return best

def ping_corrections(xa, start, boundary):
    """From one ping's two legs, estimate (data_start, delta_hz, theta0)
    using the classic symmetric-sweep CFO trick: a constant frequency
    offset shifts the up-leg and down-leg matched-filter peaks by equal
    and opposite amounts in time."""
    if boundary:
        up_guess, down_guess = start + CHIRP_LEN, start
    else:
        up_guess, down_guess = start, start + CHIRP_LEN
    up = _leg_peak(xa, up_guess, _UP_A)
    down = _leg_peak(xa, down_guess, _DOWN_A)
    if up is None or down is None:
        return None
    su, zu = up
    sd, zd = down
    shift_up   = (su - up_guess) / WORK_FS
    shift_down = (sd - down_guess) / WORK_FS
    bw = CHIRP_F1 - CHIRP_F0
    delta_hz = (shift_up - shift_down) * bw / (2 * CHIRP_DUR)
    tau      = (shift_up + shift_down) / 2.0
    # The leg search is on an integer-sample grid, so a +-1 sample jitter
    # with zero true offset still produces a nonzero delta_hz (here, one
    # sample quantum = bw/(2*CHIRP_DUR*WORK_FS) ~ 0.57 Hz). Over a long
    # high-redundancy data field a *wrong* correction this size rotates the
    # signal by several radians by the end of the frame and wrecks it, so
    # deadband out anything too small to be real Doppler/SSB drift.
    quantum = bw / (2 * CHIRP_DUR * WORK_FS)
    if abs(delta_hz) < 2.0 * quantum:
        delta_hz = 0.0
    theta0 = float(np.angle(zu)) if abs(zu) >= abs(zd) else float(np.angle(zd))
    data_start = int(round(start + 2 * CHIRP_LEN + tau * WORK_FS))
    return data_start, float(delta_hz), theta0

# ── framing: header always slot 0, ciphertext round-robins slots 1..N ───────

def pay_coded_bits(n):
    return 2 * (8 * (n + TAG_BYTES) + 6)

def n_payload_slots(n):
    return ceil(pay_coded_bits(n) / SYMS_PER_FRAME)

def frame_len_samples(L):
    return 2 * CHIRP_LEN + SYMS_PER_FRAME * L

def capacity_bytes(n_frames_avail, L):
    """Largest n (bytes) whose carousel (1 header slot + payload slots) fits
    in one lap across n_frames_avail frames."""
    def fits(n):
        return 1 + n_payload_slots(n) <= n_frames_avail

    if not fits(0):
        return -1
    lo, hi = 0, 1
    while fits(hi):
        lo, hi = hi, hi * 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo

# ── encoder ──────────────────────────────────────────────────────────────────

def encode_audio(cover, fs, message, password, redundancy="medium",
                 strength_db=DEFAULT_STRENGTH_DB, band=DEFAULT_BAND):
    if fs < WORK_FS:
        raise WavyError(f"cover sample rate {fs} Hz is below {WORK_FS} Hz")
    L = REDUNDANCY[redundancy]
    mono8 = bandpass(resample(to_mono(cover), fs, WORK_FS), band)

    n = len(message)
    nslots = n_payload_slots(n)
    total_slots = 1 + nslots
    flen = frame_len_samples(L)
    max_frames = len(mono8) // flen
    if max_frames < total_slots:
        raise WavyError(
            f"message ({n} bytes) needs >= {total_slots} frames of {flen / WORK_FS:.1f}s "
            f"at '{redundancy}' redundancy ({total_slots * flen / WORK_FS:.1f}s) but the "
            f"cover has {len(mono8) / WORK_FS:.1f}s of room ({max_frames} frames fit)")

    salt = os.urandom(SALT_BYTES)
    header = n.to_bytes(LEN_BYTES, "big") + salt
    ct = ChaCha20Poly1305(msg_key(password, salt)).encrypt(bytes(12), message, header)
    ck = carrier_key(password)

    hdr_bits = fec_encode(header)                              # exactly SYMS_PER_FRAME
    pay_bits = fec_encode(ct)
    pad = (-len(pay_bits)) % SYMS_PER_FRAME
    pay_bits = np.concatenate([pay_bits, np.zeros(pad, dtype=np.uint8)])
    slots = [hdr_bits] + [pay_bits[i * SYMS_PER_FRAME:(i + 1) * SYMS_PER_FRAME]
                          for i in range(nslots)]

    env = envelope(mono8)
    g = 10.0 ** (-strength_db / 20.0)
    g_ping = 10.0 ** (-max(strength_db - PING_BOOST_DB, 3.0) / 20.0)
    w8 = np.zeros_like(mono8)
    for k in range(max_frames):
        slot_id = k % total_slots
        pos = k * flen
        ping = ping_waveform(slot_id == 0)
        w8[pos:pos + 2 * CHIRP_LEN] += g_ping * env[pos:pos + 2 * CHIRP_LEN] * ping
        dpos = pos + 2 * CHIRP_LEN
        sym = 1.0 - 2.0 * slots[slot_id].astype(float)
        P = slot_carrier(ck, slot_id, SYMS_PER_FRAME * L, band)
        w8[dpos:dpos + SYMS_PER_FRAME * L] += g * env[dpos:dpos + SYMS_PER_FRAME * L] * np.repeat(sym, L) * P

    w = resample(w8, WORK_FS, fs)
    w = np.pad(w, (0, max(0, len(cover) - len(w))))[:len(cover)]
    out = cover + (w[:, None] if cover.ndim == 2 else w)
    return out

# ── decoder ──────────────────────────────────────────────────────────────────

def _decode_with_L(xn, xa, pings, ck, L, password, log):
    flen = frame_len_samples(L)
    accum = {}           # slot_id -> np.array soft sums, len SYMS_PER_FRAME
    counts = {}
    slot_id = None
    for start, boundary, z in pings:
        if boundary:
            slot_id = 0
        elif slot_id is None:
            continue        # haven't seen a lap boundary yet; can't number this frame safely
        else:
            slot_id += 1

        corr = ping_corrections(xa, start, boundary)
        if corr is None:
            continue
        data_start, delta_hz, theta0 = corr
        need = SYMS_PER_FRAME * L
        if data_start < 0 or data_start + need > len(xa):
            continue

        carrier_a = signal.hilbert(slot_carrier(ck, slot_id, need, (DEFAULT_BAND)))
        seg = xa[data_start:data_start + need]
        t = np.arange(need) / WORK_FS
        seg = seg * np.exp(-1j * 2 * np.pi * delta_hz * t) * np.exp(-1j * theta0)
        z_syms = (seg.reshape(SYMS_PER_FRAME, L) * np.conj(carrier_a.reshape(SYMS_PER_FRAME, L))).sum(axis=1)
        soft = z_syms.real

        if slot_id not in accum:
            accum[slot_id] = np.zeros(SYMS_PER_FRAME)
            counts[slot_id] = 0
        accum[slot_id] += soft
        counts[slot_id] += 1

    if 0 not in accum:
        return None
    header = fec_decode(accum[0])
    n = int.from_bytes(header[:LEN_BYTES], "big")
    salt = header[LEN_BYTES:HDR_BYTES]
    nslots = n_payload_slots(n)
    # A garbage header (wrong redundancy / wrong password) can claim an
    # enormous message length, which would blow up the soft-bit array and
    # Viterbi's trellis below. But a genuine message can never need more
    # payload slots than we've actually seen *some* data for — so bail the
    # instant the claim exceeds what was observed, before allocating anything.
    max_seen = max((s for s in accum if s > 0), default=0)
    if nslots < 0 or nslots > max_seen:
        return None
    pbits = pay_coded_bits(n)
    soft = np.zeros(nslots * SYMS_PER_FRAME)
    got_any = False
    for i in range(nslots):
        if (i + 1) in accum:
            soft[i * SYMS_PER_FRAME:(i + 1) * SYMS_PER_FRAME] = accum[i + 1]
            got_any = True
    if not got_any:
        return None
    ct = fec_decode(soft[:pbits])
    log(f"  L={L}: header len={n} salt={salt.hex()} slots seen={sorted(accum)}")
    try:
        return ChaCha20Poly1305(msg_key(password, salt)).decrypt(bytes(12), bytes(ct), bytes(header))
    except InvalidTag:
        return None

def decode_audio(x, fs, password, band=DEFAULT_BAND, verbose=False):
    log = (lambda s: print(s, file=sys.stderr)) if verbose else (lambda s: None)
    ck = carrier_key(password)
    r8 = bandpass(resample(to_mono(x), fs, WORK_FS), band)
    xn = r8 / envelope(r8)
    pings = find_pings(xn)
    log(f"  found {len(pings)} candidate ping(s)")
    if not pings:
        raise WavyError("no ping beacons found (wrong --band, no watermark, or signal too degraded)")
    xa = signal.hilbert(xn)
    for name in ("medium", "high", "low"):
        L = REDUNDANCY[name]
        msg = _decode_with_L(xn, xa, pings, ck, L, password, log)
        if msg is not None:
            return msg
    raise WavyError("no message found (wrong password, wrong --band, or the audio is too degraded)")

# ── CLI ──────────────────────────────────────────────────────────────────────

def get_password(confirm=False):
    pw = os.environ.get("KOMETA_PASSWORD")
    if pw:
        return pw
    pw = getpass.getpass("password: ")
    if confirm and getpass.getpass("again:    ") != pw:
        raise WavyError("passwords do not match")
    if not pw:
        raise WavyError("empty password")
    return pw

def parse_band(s):
    try:
        lo, hi = (float(v) for v in s.split("-"))
        assert 0 < lo < hi < WORK_FS / 2
        return lo, hi
    except Exception:
        raise WavyError("--band must look like 300-2700")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="wavygravy", description="ping/carousel text-in-audio steganography")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--band", default="300-2700", help="carrier band in Hz (must match on decode)")
        p.add_argument("-v", "--verbose", action="store_true")

    e = sub.add_parser("encode", help="encode <cover.wav> <message|file> <output.wav>")
    e.add_argument("cover"); e.add_argument("message"); e.add_argument("output")
    e.add_argument("--redundancy", choices=REDUNDANCY, default="medium")
    e.add_argument("--strength", type=float, default=DEFAULT_STRENGTH_DB,
                   help="dB below the cover (higher = quieter + less robust)")
    common(e)

    d = sub.add_parser("decode", help="decode <input.wav> [output.txt]")
    d.add_argument("input"); d.add_argument("output", nargs="?")
    common(d)

    c = sub.add_parser("capacity", help="capacity <cover.wav>")
    c.add_argument("cover")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "capacity":
            fs, x = read_audio(a.cover)
            n8 = int(len(to_mono(x)) * WORK_FS / fs)
            print(f"{a.cover}: {len(to_mono(x)) / fs:.1f} s @ {fs} Hz")
            for name, L in REDUNDANCY.items():
                flen = frame_len_samples(L)
                nframes = n8 // flen
                cap = capacity_bytes(nframes, L)
                print(f"  {name:<6} ({flen / WORK_FS:4.1f}s/frame, {nframes:>4} frames, "
                      f"{nframes and nframes // max(1, 1 + n_payload_slots(max(cap,0)))} lap(s)): "
                      f"{max(cap, 0)} bytes")
            return 0

        band = parse_band(a.band)
        if a.cmd == "encode":
            fs, cover = read_audio(a.cover)
            msg = open(a.message, "rb").read() if os.path.isfile(a.message) else a.message.encode("utf-8")
            pw = get_password(confirm=sys.stdin.isatty() and "KOMETA_PASSWORD" not in os.environ)
            out = encode_audio(cover, fs, msg, pw, a.redundancy, a.strength, band)
            write_audio(a.output, fs, out)
            print(f"encoded {len(msg)} bytes -> {a.output}", file=sys.stderr)
            return 0

        fs, x = read_audio(a.input)
        msg = decode_audio(x, fs, get_password(), band, a.verbose)
        if a.output:
            open(a.output, "wb").write(msg)
        else:
            sys.stdout.buffer.write(msg)
        return 0
    except WavyError as ex:
        print(f"wavygravy: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())