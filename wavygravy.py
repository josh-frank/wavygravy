#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# wavygravy.py — hide a short UTF-8 message in an audio file using self-locating
# "ping" beacons and a repeating carousel of coded chunks.
#
#   wavygravy.py encode cover.wav "stop spying on me" out.wav     (prompts for password)
#   wavygravy.py encode cover.wav - out.wav < msg.txt             ('-' stdin, '@file' file)
#   wavygravy.py decode out.wav [message.txt]
#   wavygravy.py encode --no-password cover.wav "hello" out.wav   (password-free)
#   wavygravy.py keygen hq.secret                  (optional shared secret keyfile)
#   wavygravy.py encode --keyfile hq.secret cover.wav "hi" out.wav
#   wavygravy.py encode --keyfile hq.secret --no-password ...     (keyfile-only: scriptable)
#   wavygravy.py fingerprint hq.secret             (compare by phone to confirm same file)
#   wavygravy.py capacity cover.wav
#
# Design:
#   The timeline is cut into FRAMES. Each frame starts with a PING: an up-chirp
#   and a down-chirp, in reversed order on the frame that opens each lap of the
#   carousel. Pings are found by a global matched-filter scan, so a decoder can
#   join anywhere. The up/down timing asymmetry gives a per-frame estimate of
#   frequency offset and phase, so every frame re-syncs itself (no tracking
#   loop, no frequency search). After each ping comes a DATA field: BPSK
#   symbols spread by a noise carrier that depends on the *slot number*.
#   Slot 0 carries the header (length + a random per-message salt) under a PUBLIC
#   carrier, so anyone can read it first. Slots 1..N carry the FEC'd ciphertext
#   under carriers keyed by the password, round-robin, lap after lap. The decoder
#   sums soft values across laps before Viterbi, so gaps self-heal on the next lap.
#
# Keys: one scrypt call (N=2^17) over (password, that message's salt, optional
# keyfile) gives a master key; the payload carriers and the AEAD key/nonce come
# from it by HKDF. The encryption is ChaCha20-Poly1305. Because the salt is
# per-message, password guesses can't be precomputed across files and no two
# messages share carriers. An optional keyfile (32 random bytes, `keygen`) is
# mixed into the scrypt salt: with it, stolen audio can't be password-guessed at
# all. Losing the keyfile loses every message made with it.
#
# Known trade-offs: chirps are more recognizable in a spectrogram than keyed
# noise would be, and are public. The header is public too (the message length,
# which the lap length already gives away to within a slot, and a random salt).
# The password protects the content, not the fact that something is hidden.
# ─────────────────────────────────────────────────────────────────────────────

import argparse, hashlib, os, sys, getpass, subprocess, tempfile
from math import gcd, ceil

import numpy as np
from scipy import signal
from scipy.io import wavfile
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

# ── constants ────────────────────────────────────────────────────────────────

WORK_FS       = 8000
DEFAULT_BAND  = (300.0, 2700.0)
KDF_N         = 2 ** 17                       # scrypt: 128 MiB, ~0.4 s per password guess
KDF_MAXMEM    = 256 * 2 ** 20                 # hashlib's default cap (32 MiB) is too small
KDF_LABEL     = b"wavygravy/master/v3"
HEADER_KEY    = hashlib.sha256(b"wavygravy/public-header/v3").digest()   # slot 0 carrier: public
KEY_BYTES     = 32                            # keyfile size

CHIRP_DUR     = 0.25                          # seconds, each leg of a ping
CHIRP_LEN     = int(CHIRP_DUR * WORK_FS)
CHIRP_F0      = 350.0                         # sweep range, inside DEFAULT_BAND
CHIRP_F1      = 2650.0
PING_Z        = 7.0                           # ping-detect threshold (std devs)
# Pings ride louder than data: at equal level they barely beat the cover's own
# chance correlation with the chirp. A lost ping costs one slot for one lap;
# lost sync costs the whole message.
PING_BOOST_DB = 6.0
LEG_SEARCH    = 40                            # samples, fine-align search window

REDUNDANCY    = {"low": 128, "medium": 256, "high": 512}   # samples per data bit
LEN_BYTES     = 2
SALT_BYTES    = 8
MAX_MESSAGE_BYTES = 16 * 1024
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

def _pw(password):
    """None and "" both mean 'no password': keys derive from an empty string,
    i.e. they are public. The message is still FEC'd and integrity-checked,
    but anyone with this tool can read it."""
    return b"" if password is None else password.encode()

def master_key(password, salt, secret=None):
    """The only slow step; everything else is cheap HKDF off this. `salt` is the
    random per-message salt read from the public header, so a password guess
    costs one scrypt *per file* and can't be precomputed across files. With a
    keyfile (`secret`) the scrypt salt also contains a secret: without it, no
    password guess can even be tested."""
    if secret is not None and len(secret) != KEY_BYTES:
        raise WavyError(f"keyfile secret must be {KEY_BYTES} bytes")
    sec = b"" if secret is None else secret
    kdf_salt = hashlib.sha256(KDF_LABEL + bytes([len(sec)]) + sec + salt).digest()
    return hashlib.scrypt(_pw(password), salt=kdf_salt, n=KDF_N, r=8, p=1,
                          dklen=32, maxmem=KDF_MAXMEM)

def fingerprint(secret):
    """Short non-secret ID ('xxxx-xxxx-xxxx') for comparing keyfiles by phone."""
    h = hashlib.sha256(b"wavygravy/fingerprint/v1" + secret).hexdigest()[:12]
    return "-".join(h[i:i + 4] for i in range(0, 12, 4))

def load_keyfile(path):
    """Read a keyfile written by keygen: exactly 64 hex characters. Anything else
    (a typed word, a binary file) is refused, so a weak keyfile can't be made by
    accident. On POSIX, refuses files other users can read, as ssh does."""
    bad = WavyError(f"{path} is not a wavygravy keyfile (expected 64 hex characters; "
                    "make one with: wavygravy keygen NAME.secret)")
    try:
        if os.name == "posix" and os.stat(path).st_mode & 0o077:
            raise WavyError(f"keyfile {path} is accessible by other users; run: chmod 600 {path}")
        with open(path, "rb") as f:
            raw = f.read(200)
    except OSError as ex:
        raise WavyError(f"cannot read keyfile: {ex}")
    try:
        txt = raw.decode("ascii").strip()
        key = bytes.fromhex(txt)
    except ValueError:                                    # includes UnicodeDecodeError
        raise bad
    if len(txt) != 2 * KEY_BYTES:                         # fromhex tolerates spaces
        raise bad
    return key

def keygen(path):
    """Write a fresh random keyfile (mode 0600, never overwrites); returns its fingerprint."""
    key = os.urandom(KEY_BYTES)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise WavyError(f"{path} already exists (refusing to overwrite a keyfile)")
    except OSError as ex:
        raise WavyError(f"cannot create keyfile: {ex}")
    with os.fdopen(fd, "w") as f:
        f.write(key.hex() + "\n")
    return fingerprint(key)

def derive(master, info, length=32, salt=None):
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(master)

def keystream(key32, nbytes):
    enc = Cipher(algorithms.ChaCha20(key32, bytes(16)), mode=None).encryptor()
    return enc.update(bytes(nbytes))

def keyed_bits(key32, n):
    ks = np.frombuffer(keystream(key32, (n + 7) // 8), dtype=np.uint8)
    return np.unpackbits(ks)[:n]

def msg_key_nonce(master):
    """Per-message AEAD key (32 B) and nonce (12 B). `master` already contains the
    message's random salt, so (key, nonce) repeats only if the salt does: it
    must stay random (os.urandom), never shortened or reused."""
    km = derive(master, b"wavygravy/aead", 44)
    return km[:32], km[32:]

# ── DSP helpers ──────────────────────────────────────────────────────────────

def bp_fir(band=DEFAULT_BAND):
    return signal.firwin(401, band, pass_zero=False, fs=WORK_FS)

def bandpass(x, band=DEFAULT_BAND):
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

def slot_carrier(mk, slot, n, band=DEFAULT_BAND):
    """Band-limited noise carrier for one slot, unit RMS, from key material `mk`
    (HEADER_KEY for slot 0, the master key for payload slots). Keyed by slot
    number (not sample position) so a decoder joining mid-stream can despread
    a slot as soon as it knows its number."""
    key = derive(mk, b"wavygravy/pn/" + str(int(slot)).encode())
    chips = 1.0 - 2.0 * keyed_bits(key, n)
    return bandpass(chips, band) / np.sqrt(np.sum(bp_fir(band) ** 2))

# ── audio I/O ────────────────────────────────────────────────────────────────

def _to_float(data):
    # float32 halves the footprint of the full-rate input; everything is
    # promoted to float64 at 8 kHz, where the arrays are small.
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max + 1)
    return data.astype(np.float32)

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

def fec_encode(data):
    """bytes -> coded bits (rate 1/2, 6 tail bits)."""
    bits = np.concatenate([np.unpackbits(np.frombuffer(data, dtype=np.uint8)),
                           np.zeros(6, np.uint8)])
    out, state = [], 0
    for u in bits:
        reg = (int(u) << 6) | state
        out += [_parity(reg & _G1), _parity(reg & _G2)]
        state = reg >> 1
    return np.array(out, dtype=np.uint8)

def fec_decode(soft):
    """soft coded values -> bytes."""
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
    return np.packbits(bits[:-6]).tobytes()

# ── pings: a chirp pair whose leg order marks carousel-lap boundaries ───────

def _make_chirp(f0, f1, n):
    t = np.arange(n) / WORK_FS
    k = (f1 - f0) / (n / WORK_FS)
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))

_UP   = _make_chirp(CHIRP_F0, CHIRP_F1, CHIRP_LEN)
_DOWN = _make_chirp(CHIRP_F1, CHIRP_F0, CHIRP_LEN)
_UP_A, _DOWN_A = signal.hilbert(_UP), signal.hilbert(_DOWN)   # analytic refs

def ping_waveform(boundary):
    """boundary=True marks the frame that opens a carousel lap (slot 0)."""
    return np.concatenate([_DOWN, _UP]) if boundary else np.concatenate([_UP, _DOWN])

def find_pings(xn, z_thresh=PING_Z):
    """Global matched-filter scan, no assumption about where the stream
    starts. Returns [(start_sample, boundary_flag, z), ...] time-ordered."""
    cn = signal.fftconvolve(xn, ping_waveform(False)[::-1], mode="valid")
    cb = signal.fftconvolve(xn, ping_waveform(True)[::-1], mode="valid")
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
    """From one ping's two legs, estimate (data_start, delta_hz, theta0) with
    the symmetric-sweep trick: a constant frequency offset shifts the up-leg
    and down-leg peaks by equal and opposite amounts in time."""
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
    # The integer-sample leg search makes +-1 sample jitter look like a
    # ~0.57 Hz offset; over a long data field a wrong correction that size
    # rotates the signal by radians. Deadband anything below 2 quanta.
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

def capacity_bytes(n_frames_avail, L=None):
    """Largest message (bytes) whose carousel (1 header slot + payload slots)
    fits in one lap of n_frames_avail frames; -1 if even the header won't.
    (L is unused: the slot count doesn't depend on redundancy. Kept so existing
    callers keep working.)"""
    return max(-1, ((n_frames_avail - 1) * SYMS_PER_FRAME // 2 - 6) // 8 - TAG_BYTES)

# ── encoder ──────────────────────────────────────────────────────────────────

def encode_audio(cover, fs, message, password, redundancy="medium",
                 strength_db=DEFAULT_STRENGTH_DB, band=DEFAULT_BAND, secret=None):
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
    mk = master_key(password, salt, secret)
    header = n.to_bytes(LEN_BYTES, "big") + salt
    key, nonce = msg_key_nonce(mk)
    ct = ChaCha20Poly1305(key).encrypt(nonce, message, header)

    hdr_bits = fec_encode(header)                             # exactly SYMS_PER_FRAME
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
        P = slot_carrier(HEADER_KEY if slot_id == 0 else mk, slot_id, SYMS_PER_FRAME * L, band)
        w8[dpos:dpos + SYMS_PER_FRAME * L] += g * env[dpos:dpos + SYMS_PER_FRAME * L] * np.repeat(sym, L) * P

    w = resample(w8, WORK_FS, fs)
    w = np.pad(w, (0, max(0, len(cover) - len(w))))[:len(cover)]
    w = w.astype(cover.dtype)
    return cover + (w[:, None] if cover.ndim == 2 else w)

# ── decoder ──────────────────────────────────────────────────────────────────

def _frames(xa, pings, L, log):
    """Password-independent sync for every usable ping: [(slot_id, data_start,
    delta_hz, theta0), ...]. Slot numbers count pings up from each lap boundary."""
    frames, slot_id, need = [], None, SYMS_PER_FRAME * L
    for start, boundary, z in pings:
        if boundary:
            slot_id = 0
        elif slot_id is None:
            continue        # no lap boundary seen yet; can't number this frame
        else:
            slot_id += 1

        corr = ping_corrections(xa, start, boundary)
        if corr is None:
            # Leg alignment failed (more drift than LEG_SEARCH tolerates):
            # skip this frame; the carousel picks it up next lap.
            log(f"  slot {slot_id}: no leg alignment near sample {start}, skipping this lap")
            continue
        data_start, delta_hz, theta0 = corr
        if data_start < 0 or data_start + need > len(xa):
            log(f"  slot {slot_id}: data field at sample {data_start} runs off the end, skipping")
            continue
        frames.append((slot_id, data_start, delta_hz, theta0))
    return frames

def _despread(xa, frame, key, L, band):
    """Soft symbol values for one frame's data field under carrier key material `key`."""
    slot_id, data_start, delta_hz, theta0 = frame
    need = SYMS_PER_FRAME * L
    carrier_a = signal.hilbert(slot_carrier(key, slot_id, need, band))
    t = np.arange(need) / WORK_FS
    seg = xa[data_start:data_start + need]
    seg = seg * np.exp(-1j * 2 * np.pi * delta_hz * t) * np.exp(-1j * theta0)
    z_syms = (seg.reshape(SYMS_PER_FRAME, L) * np.conj(carrier_a.reshape(SYMS_PER_FRAME, L))).sum(axis=1)
    return z_syms.real

def _decode_with_L(xa, pings, password, secret, L, band, log):
    frames = _frames(xa, pings, L, log)

    # Pass 1: the header sits under a public carrier, so it needs no key.
    hdr = [_despread(xa, f, HEADER_KEY, L, band) for f in frames if f[0] == 0]
    if not hdr:
        return None
    header = fec_decode(sum(hdr))
    n = int.from_bytes(header[:LEN_BYTES], "big")
    salt = header[LEN_BYTES:HDR_BYTES]
    # A garbage header (wrong redundancy level) can claim an absurd length.
    # Reject it by a flat cap, then by the real bound: a genuine message never
    # needs more payload slots than were actually observed in this recording.
    # (This also keeps the slow scrypt below from running on garbage.)
    if n > MAX_MESSAGE_BYTES:
        return None
    nslots = n_payload_slots(n)
    seen = sorted({f[0] for f in frames if f[0] > 0})
    if nslots > (seen[-1] if seen else 0):
        return None
    log(f"  L={L}: header len={n} salt={salt.hex()} slots seen={[0] + seen}")

    # Pass 2: payload carriers are keyed by scrypt(password, this file's salt, keyfile).
    mk = master_key(password, salt, secret)
    soft = np.zeros(nslots * SYMS_PER_FRAME)
    got_any = False
    for f in frames:
        if 0 < f[0] <= nslots:
            soft[(f[0] - 1) * SYMS_PER_FRAME:f[0] * SYMS_PER_FRAME] += _despread(xa, f, mk, L, band)
            got_any = True
    if not got_any:
        return None
    ct = fec_decode(soft[:pay_coded_bits(n)])
    try:
        key, nonce = msg_key_nonce(mk)
        return ChaCha20Poly1305(key).decrypt(nonce, ct, header)
    except InvalidTag:
        return None

def decode_audio(x, fs, password, band=DEFAULT_BAND, verbose=False, secret=None):
    log = (lambda s: print(s, file=sys.stderr)) if verbose else (lambda s: None)
    r8 = bandpass(resample(to_mono(x), fs, WORK_FS), band)
    xn = r8 / envelope(r8)
    pings = find_pings(xn)
    log(f"  found {len(pings)} candidate ping(s)")
    if not pings:
        raise WavyError(f"no ping beacons found in band {band[0]:.0f}-{band[1]:.0f} Hz "
                        "(wrong --band? this file may have none, or the signal is too degraded)")
    xa = signal.hilbert(xn)
    for name in ("medium", "high", "low"):
        msg = _decode_with_L(xa, pings, password, secret, REDUNDANCY[name], band, log)
        if msg is not None:
            return msg
    # Pings were found, so the remaining explanations are a wrong password or
    # keyfile, or payload-level damage.
    raise WavyError("pings found but no message decoded (wrong password or keyfile, "
                    "or the payload is too degraded)")

# ── CLI ──────────────────────────────────────────────────────────────────────

def get_password(confirm=False, no_password=False):
    """Interactive prompt only. There is deliberately no env-var or argv route:
    both leak to other processes. For unattended use, use a keyfile with
    --no-password."""
    if no_password:
        return ""
    pw = getpass.getpass("password: ")
    if confirm and getpass.getpass("again:    ") != pw:
        raise WavyError("passwords do not match")
    if not pw:
        raise WavyError("empty password (use --no-password for password-free mode)")
    return pw

def keyfile_from_args(a):
    if not a.keyfile:
        return None
    secret = load_keyfile(a.keyfile)
    if a.verbose:
        print(f"  keyfile fingerprint {fingerprint(secret)}", file=sys.stderr)
    return secret

def read_message(arg):
    """'-' reads stdin and '@path' reads a file, both of which keep the message
    out of shell history and `ps`. Anything else is the literal message."""
    try:
        if arg == "-":
            return sys.stdin.buffer.read()
        if arg.startswith("@"):
            with open(arg[1:], "rb") as f:
                return f.read()
    except OSError as ex:
        raise WavyError(f"cannot read message: {ex}")
    return arg.encode("utf-8")

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
        p.add_argument("--keyfile", metavar="FILE.secret",
                       help="secret keyfile from `keygen` (must match on decode)")
        p.add_argument("--no-password", action="store_true",
                       help="no password (must match on decode). With --keyfile: keyfile-only. "
                            "Without: anyone with this tool can decode")

    e = sub.add_parser("encode", help="encode <cover.wav> <message|-|@file> <output.wav>")
    e.add_argument("cover")
    e.add_argument("message", help="literal text (visible in shell history and `ps`), "
                   "'-' to read stdin, or '@path' to read a file")
    e.add_argument("output")
    e.add_argument("--redundancy", choices=REDUNDANCY, default="medium")
    e.add_argument("--strength", type=float, default=DEFAULT_STRENGTH_DB,
                   help="dB below the cover (higher = quieter + less robust)")
    common(e)

    d = sub.add_parser("decode", help="decode <input.wav> [output.txt]")
    d.add_argument("input"); d.add_argument("output", nargs="?")
    common(d)

    c = sub.add_parser("capacity", help="capacity <cover.wav>")
    c.add_argument("cover")

    k = sub.add_parser("keygen", help="keygen <NAME.secret>: create a random keyfile")
    k.add_argument("path")
    f = sub.add_parser("fingerprint", help="fingerprint <NAME.secret>: ID to compare by phone")
    f.add_argument("path")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "keygen":
            fp = keygen(a.path)
            print(f"created {a.path}   fingerprint {fp}\n"
                  "Give the other party a copy over a trusted channel and compare fingerprints by phone.\n"
                  "Back it up: if it is lost, every message made with it is unrecoverable.", file=sys.stderr)
            return 0

        if a.cmd == "fingerprint":
            print(fingerprint(load_keyfile(a.path)))
            return 0

        if a.cmd == "capacity":
            fs, x = read_audio(a.cover)
            n8 = int(len(to_mono(x)) * WORK_FS / fs)
            print(f"{a.cover}: {len(to_mono(x)) / fs:.1f} s @ {fs} Hz")
            for name, L in REDUNDANCY.items():
                flen = frame_len_samples(L)
                nframes = n8 // flen
                cap = capacity_bytes(nframes)
                head = f"  {name:<6} ({flen / WORK_FS:4.1f}s/frame, {nframes:>4} frames"
                if cap < 0:
                    print(f"{head}): too short even for the header (needs >= 1 frame)")
                else:
                    laps = nframes // (1 + n_payload_slots(cap))
                    print(f"{head}, {laps} lap(s)): {cap} bytes")
            return 0

        if a.cmd == "encode":
            fs, cover = read_audio(a.cover)
            msg = read_message(a.message)
            secret = keyfile_from_args(a)             # before the prompt: fail fast on a bad file
            pw = get_password(confirm=True, no_password=a.no_password)
            write_audio(a.output, fs, encode_audio(cover, fs, msg, pw, a.redundancy, a.strength,
                                                    parse_band(a.band), secret))
            note = ""
            if a.no_password:
                note = (" (keyfile-only: no password)" if secret
                        else " (no password: readable by anyone with this tool)")
            print(f"encoded {len(msg)} bytes -> {a.output}{note}", file=sys.stderr)
            return 0

        fs, x = read_audio(a.input)
        secret = keyfile_from_args(a)
        msg = decode_audio(x, fs, get_password(no_password=a.no_password), parse_band(a.band),
                           a.verbose, secret)
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