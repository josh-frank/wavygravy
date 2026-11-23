#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# wavygravy.py — hide a short UTF-8 message in an audio file, recoverable
# after telephone / HAM-radio grade degradation.  (first draft)
#
#   WAVYGRAVY_PASSWORD=... wavygravy.py encode  cover.wav "stop spying on me" out.wav
#   WAVYGRAVY_PASSWORD=... wavygravy.py decode  out.wav [message.txt]
#                       wavygravy.py capacity cover.wav
#
# If WAVYGRAVY_PASSWORD is unset the password is prompted for (not echoed).
#
# How it works (see the design notes at the bottom of the file):
#   message ─ChaCha20-Poly1305─▶ frame ─conv. code r=1/2 + interleave─▶ bits
#   bits ─BPSK─▶ × keyed 300-2700 Hz noise carrier (512 samples per bit @ 8 kHz)
#        ─▶ scaled to follow the cover's loudness, ~18 dB below it ─▶ cover + wm
#   Keyed sync words are interleaved every 56 data bits so the decoder can find
#   the signal and track clock drift.  The decoder correlates (I and Q) against
#   the same keyed carrier, then runs a soft-decision Viterbi decoder.
# ─────────────────────────────────────────────────────────────────────────────

import argparse, hashlib, hmac, os, sys, getpass, subprocess, tempfile
from math import gcd, ceil

import numpy as np
import scipy.fft as sfft
from scipy import signal
from scipy.io import wavfile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

# ── constants ────────────────────────────────────────────────────────────────

WORK_FS      = 8000                       # internal rate (telephone rate)
SYNC_BITS    = 8                          # keyed sync symbols per block
DATA_BITS    = 56                         # coded data symbols per block
BLOCK        = SYNC_BITS + DATA_BITS
REDUNDANCY   = {"low": 256, "medium": 512, "high": 1024}   # samples per bit
DEFAULT_BAND = (300.0, 2700.0)
DEFAULT_STRENGTH_DB = 18.0                # watermark this far below the cover
LEN_BYTES    = 8                          # length field width (big-endian uint)
SALT_BYTES   = 8
HDR_BYTES    = LEN_BYTES + SALT_BYTES     # len(8) + salt(8)
TAG_BYTES    = 16
ENV_WIN      = 400                        # 50 ms @ 8 kHz
ENV_FLOOR    = 0.1                        # envelope floor, fraction of global RMS
SYNC_Z       = 7.0                        # sync detection threshold (std devs)
FREQ_STEP    = 2.5                        # Hz, decoder frequency-offset grid
KDF_N        = 2 ** 14

class WAVyGravyError(Exception):
    pass

# ── keys & keyed pseudo-randomness ───────────────────────────────────────────

def carrier_key(password):
    """Key for the carrier / sync / interleaver. Depends on the password only,
    because the decoder needs it *before* it can read anything from the file."""
    return hashlib.scrypt(password.encode(), salt=b"wavygravy/carrier/v1",
                          n=KDF_N, r=8, p=1, dklen=32)

def subkey(ck, label):
    return hmac.new(ck, label, hashlib.sha256).digest()

def keystream(key32, nbytes):
    enc = Cipher(algorithms.ChaCha20(key32, bytes(16)), mode=None).encryptor()
    return enc.update(bytes(nbytes))

def keyed_bits(key32, n):
    ks = np.frombuffer(keystream(key32, (n + 7) // 8), dtype=np.uint8)
    return np.unpackbits(ks)[:n]

def keyed_perm(ck, label, n):
    """Deterministic permutation (argsort of keystream words; stable across
    Python/numpy versions, unlike random.shuffle / Generator.permutation)."""
    ks = np.frombuffer(keystream(subkey(ck, label), 4 * n), dtype="<u4")
    return np.argsort(ks, kind="stable")

def sync_word(ck):
    return 1.0 - 2.0 * keyed_bits(subkey(ck, b"sync"), SYNC_BITS)

# ── DSP helpers ──────────────────────────────────────────────────────────────

def bp_fir(band):
    return signal.firwin(401, band, pass_zero=False, fs=WORK_FS)

def bandpass(x, band):
    # FIR + 'same' gives results independent of signal length (needed so the
    # encoder's and decoder's carriers agree sample-for-sample).
    return signal.fftconvolve(x, bp_fir(band), mode="same")

def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    g = gcd(int(fs_in), int(fs_out))
    return signal.resample_poly(x, int(fs_out) // g, int(fs_in) // g)

def envelope(xb):
    """Short-term RMS of a band-limited signal, floored at ENV_FLOOR x global."""
    m = np.sqrt(np.maximum(
        signal.fftconvolve(xb ** 2, np.ones(ENV_WIN) / ENV_WIN, mode="same"), 0))
    return np.maximum(m, ENV_FLOOR * (np.sqrt(np.mean(xb ** 2)) + 1e-12))

def carrier(ck, n, band):
    """Keyed band-limited noise carrier, unit RMS."""
    chips = 1.0 - 2.0 * keyed_bits(subkey(ck, b"pn"), n)
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
        # not a plain PCM WAV → let ffmpeg convert it
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, "in.wav")
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path,
                                "-f", "wav", "-acodec", "pcm_s16le", tmp],
                               check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError):
                raise WAVyGravyError(f"cannot read {path} (not a PCM WAV, and ffmpeg could not convert it)")
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

# trellis tables: state = previous 6 input bits; reg = (u << 6) | state
_NS     = np.arange(64)
_PRED   = np.stack([((_NS & 31) << 1) | 0, ((_NS & 31) << 1) | 1])   # (2, 64)
_INBIT  = _NS >> 5
_EXP    = np.zeros((2, 64, 2))                                       # ±1 expected outputs
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
    """soft: float array, + means bit 0, − means bit 1, 0 = erasure.
    Returns info bits (tail removed). Trellis is terminated in state 0."""
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

def fec_encode(data, ck, label):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    coded = conv_encode(bits)
    return coded[keyed_perm(ck, label, len(coded))]

def fec_decode(soft, ck, label):
    perm = keyed_perm(ck, label, len(soft))
    deint = np.empty_like(soft); deint[perm] = soft
    bits = viterbi_decode(deint)
    return np.packbits(bits).tobytes()

# ── framing ──────────────────────────────────────────────────────────────────

def hdr_coded_bits():
    return 2 * (8 * HDR_BYTES + 6)

def pay_coded_bits(n):
    return 2 * (8 * (n + TAG_BYTES) + 6)

def plan(n):
    """(data_bits, blocks, positions) for an n-byte message."""
    data = hdr_coded_bits() + pay_coded_bits(n)
    blocks = ceil(data / DATA_BITS)
    return data, blocks, data + SYNC_BITS * blocks

def capacity_bytes(n_samples_8k, L):
    """Largest n (bytes) such that plan(n)'s symbol count fits in n_samples_8k
    samples at L samples/bit. positions(n) is strictly increasing in n (it
    grows by ~16 coded bits per payload byte), so exponential search for an
    upper bound followed by binary search finds it in O(log n) calls to
    plan(), with no fixed ceiling on message size."""
    def fits(n):
        return plan(n)[2] * L <= n_samples_8k

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

def msg_key(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=KDF_N, r=8, p=1, dklen=32)

def build_symbols(data_bits, sw):
    """Interleave sync words and data into BPSK symbols (±1)."""
    chunks = []
    for b in range(ceil(len(data_bits) / DATA_BITS)):
        chunks.append(sw)
        chunks.append(1.0 - 2.0 * data_bits[b * DATA_BITS:(b + 1) * DATA_BITS].astype(float))
    return np.concatenate(chunks)

# ── encoder ──────────────────────────────────────────────────────────────────

def encode_audio(cover, fs, message, password, redundancy="medium",
                 strength_db=DEFAULT_STRENGTH_DB, band=DEFAULT_BAND):
    if fs < WORK_FS:
        raise WAVyGravyError(f"cover sample rate {fs} Hz is below {WORK_FS} Hz")
    L = REDUNDANCY[redundancy]
    mono8 = bandpass(resample(to_mono(cover), fs, WORK_FS), band)

    n = len(message)
    _, _, positions = plan(n)
    need = positions * L
    if need > len(mono8):
        cap = capacity_bytes(len(mono8), L)
        raise WAVyGravyError(
            f"message ({n} bytes) needs {need / WORK_FS:.1f} s of cover at '{redundancy}' "
            f"redundancy but the cover has {len(mono8) / WORK_FS:.1f} s "
            f"(capacity here: {max(cap, 0)} bytes)")

    # frame: [len][salt] in the clear (but FEC'd), then ciphertext+tag
    salt = os.urandom(SALT_BYTES)
    header = n.to_bytes(LEN_BYTES, "big") + salt
    ct = ChaCha20Poly1305(msg_key(password, salt)).encrypt(bytes(12), message, header)
    ck = carrier_key(password)
    data_bits = np.concatenate([fec_encode(header, ck, b"perm-hdr"),
                                fec_encode(ct, ck, b"perm-pay")])
    sym = build_symbols(data_bits, sync_word(ck))
    assert len(sym) == positions

    # spread, shape, add
    P = carrier(ck, need, band)
    env = envelope(mono8)[:need]
    g = 10.0 ** (-strength_db / 20.0)
    w8 = g * env * np.repeat(sym, L) * P
    w = resample(w8, WORK_FS, fs)
    w = np.pad(w, (0, max(0, len(cover) - len(w))))[:len(cover)]
    out = cover + (w[:, None] if cover.ndim == 2 else w)
    return out

# ── decoder ──────────────────────────────────────────────────────────────────

class _Demod:
    """Block-by-block demodulator: sync re-lock (timing/drift), I/Q despreading,
    and a second-order decision-directed phase/frequency tracking loop."""

    def __init__(self, xs, pa, sw, L, t0):
        self.xs, self.pa, self.sw, self.L = xs, pa, sw, L
        self.t0 = float(t0)
        self.b = 0
        self.anchors = []           # (block index, refined start sample)
        self.eps = 0.0              # clock-drift estimate (fractional)
        self.theta = 0.0            # tracked carrier phase
        self.omega = 0.0            # tracked phase rotation per bit
        self.soft = []
        self.sync_q = []

    def _z(self, start, p):
        L = self.L
        if start < 0 or start + L > len(self.xs) or (p + 1) * L > len(self.pa):
            return None
        return np.dot(self.xs[start:start + L], self.pa[p * L:(p + 1) * L])

    def _sync_metric(self, tau, b):
        L, tot = self.L, 0.0
        for k in range(SYNC_BITS):
            z = self._z(int(round(tau + k * L * (1 + self.eps))), b * BLOCK + k)
            if z is None:
                return -1.0
            tot += abs(z) ** 2
        return tot

    def _relock(self, b, W=4):
        if b == 0:
            pred = self.t0
        else:
            pred = self.anchors[-1][1] + (b - self.anchors[-1][0]) * BLOCK * self.L * (1 + self.eps)
        c = int(round(pred))
        taus = np.arange(c - W, c + W + 1)
        m = np.array([self._sync_metric(t, b) for t in taus])
        i = int(np.argmax(m))
        s = float(taus[i])
        if 0 < i < len(taus) - 1 and m[i] > 0:      # parabolic sub-sample refinement
            d = m[i - 1] - 2 * m[i] + m[i + 1]
            if d < 0:
                s += 0.5 * (m[i - 1] - m[i + 1]) / d
        self.anchors.append((b, s))
        if len(self.anchors) >= 2:
            bs = np.array([a[0] for a in self.anchors], float) * BLOCK * self.L
            ss = np.array([a[1] for a in self.anchors], float)
            self.eps = float(np.clip(np.polyfit(bs, ss, 1)[0] - 1.0, -3e-3, 3e-3))
        return s

    def _init_phase(self, zs):
        u = np.array(zs) * self.sw
        ang = np.unwrap(np.angle(u))
        k = np.arange(len(u))
        slope, icpt = np.polyfit(k, ang, 1, w=np.abs(u) + 1e-12)
        self.omega, self.theta = float(slope), float(icpt) - float(slope)

    def _block(self):
        b, L = self.b, self.L
        s = self._relock(b)
        zs = []
        for k in range(BLOCK):
            zs.append(self._z(int(round(s + k * L * (1 + self.eps))), b * BLOCK + k))
        if b == 0 and all(z is not None for z in zs[:SYNC_BITS]):
            self._init_phase(zs[:SYNC_BITS])
        alpha, beta = 0.10, 0.003
        for k, z in enumerate(zs):
            if z is None:
                if k >= SYNC_BITS:
                    self.soft.append(0.0)
                continue
            self.theta += self.omega
            u = z * np.exp(-1j * self.theta)
            if k < SYNC_BITS:
                e = np.angle(u * self.sw[k])
                self.sync_q.append(u.real * self.sw[k])
            else:
                e = np.arctan2(u.imag * (1.0 if u.real >= 0 else -1.0), abs(u.real))
                self.soft.append(float(u.real))
            self.theta += alpha * e
            self.omega += beta * e
        self.b += 1

    def get(self, nbits):
        while len(self.soft) < nbits:
            if (self.b * BLOCK) * self.L + BLOCK * self.L // 2 > len(self.xs):
                break
            self._block()
        s = np.array(self.soft[:nbits])
        if len(s) < nbits:
            s = np.concatenate([s, np.zeros(nbits - len(s))])
        return s


def _sync_candidates(xn, pa, sw, L, max_freq, log):
    """Yield (delta_hz, lag, z) sync candidates, best-first per frequency region."""
    N = len(xn)
    if N < SYNC_BITS * L * 2:
        return
    M = sfft.next_fast_len(N)
    refs = [np.conj(sfft.fft(np.conj(pa[k * L:(k + 1) * L]), M)) for k in range(SYNC_BITS)]
    T = N - SYNC_BITS * L + 1
    analytic = None
    n_idx = np.arange(N)

    def evaluate(delta):
        nonlocal analytic
        if delta == 0:
            xs = xn
        else:
            if analytic is None:
                analytic = signal.hilbert(xn)
            xs = (analytic * np.exp(-2j * np.pi * delta * n_idx / WORK_FS)).real
        X = sfft.fft(xs, M)
        metric = np.zeros(T)
        for k in range(SYNC_BITS):
            c = sfft.ifft(X * refs[k])
            metric += np.abs(c[k * L:k * L + T]) ** 2
        i = int(np.argmax(metric))
        z = (metric[i] - metric.mean()) / (metric.std() + 1e-30)
        return xs, i, float(z)

    steps = int(round(max_freq / FREQ_STEP))
    grid = sorted({round(j * FREQ_STEP, 3) for j in range(-steps, steps + 1)}, key=abs)
    cache, done = {}, set()
    for d in grid:
        if d in done:
            continue
        cache[d] = evaluate(d); done.add(d)
        log(f"  L={L} δ={d:+.1f} Hz: sync z={cache[d][2]:.1f}")
        if cache[d][2] >= SYNC_Z:
            for dd in (d - 5, d - 2.5, d + 2.5, d + 5):            # refine neighbourhood
                if abs(dd) <= max_freq + 1e-9 and dd not in done:
                    cache[dd] = evaluate(dd); done.add(dd)
            best = max((x for x in cache if abs(x - d) <= 5 + 1e-9), key=lambda x: cache[x][2])
            xs, lag, z = cache[best]
            yield best, xs, lag, z


def _try(xs, pa, sw, L, lag, ck, password, log):
    dm = _Demod(xs, pa, sw, L, lag)
    hdr = fec_decode(dm.get(hdr_coded_bits()), ck, b"perm-hdr")
    n, salt = int.from_bytes(hdr[:LEN_BYTES], "big"), hdr[LEN_BYTES:HDR_BYTES]
    # A wrong sync candidate (wrong password/offset/frequency) can decode the
    # header to a garbage length. With a single-byte length field that used to
    # be capped at 255; now it can be astronomically large, and
    # pay_coded_bits(n) would then ask for an impossibly large soft-bit array.
    # Bound n by what could possibly still fit in the remaining audio and bail
    # out early (as just another failed candidate) rather than crash.
    max_bits = (max(0, len(xs) - lag) // L + 1) * DATA_BITS
    if pay_coded_bits(n) > max_bits:
        return None
    soft = dm.get(hdr_coded_bits() + pay_coded_bits(n))[hdr_coded_bits():]
    ct = fec_decode(soft, ck, b"perm-pay")
    if dm.anchors:
        log(f"    lag={lag} len={n} drift={dm.eps * 1e6:+.0f} ppm blocks={dm.b}")
    try:
        return ChaCha20Poly1305(msg_key(password, bytes(salt))).decrypt(bytes(12), bytes(ct), bytes(hdr))
    except InvalidTag:
        return None


def decode_audio(x, fs, password, band=DEFAULT_BAND, max_freq=30.0, verbose=False):
    log = (lambda s: print(s, file=sys.stderr)) if verbose else (lambda s: None)
    ck = carrier_key(password)
    sw = sync_word(ck)
    r8 = bandpass(resample(to_mono(x), fs, WORK_FS), band)
    xn = r8 / envelope(r8)                        # AGC: gain-invariant, flattens cover loudness
    pa = signal.hilbert(carrier(ck, len(xn), band))
    for name in ("medium", "high", "low"):
        L = REDUNDANCY[name]
        for delta, xs, lag, z in _sync_candidates(xn, pa, sw, L, max_freq, log):
            log(f"  candidate: L={L} δ={delta:+.1f} Hz lag={lag} z={z:.1f}")
            msg = _try(xs, pa, sw, L, lag, ck, password, log)
            if msg is not None:
                return msg
    raise WAVyGravyError("no message found (wrong password, wrong --band, or the audio is too degraded)")

# ── channel simulation (for --simulate) ──────────────────────────────────────

def simulate_telephone(x, fs, snr_db=30.0, seed=0):
    """300-3400 Hz band limit, 8 kHz, 8-bit µ-law, light noise. Returns (y, 8000)."""
    y = resample(to_mono(x), fs, 8000)
    y = signal.fftconvolve(y, signal.firwin(257, [300, 3400], pass_zero=False, fs=8000), mode="same")
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round(c * 127) / 127
    y = np.sign(c) * (np.expm1(np.abs(c) * np.log1p(mu))) / mu
    y = y + np.random.default_rng(seed).normal(0, np.sqrt(np.mean(y ** 2)) * 10 ** (-snr_db / 20), len(y))
    return y, 8000

# ── CLI ──────────────────────────────────────────────────────────────────────

def get_password(confirm=False):
    pw = os.environ.get("WAVYGRAVY_PASSWORD")
    if pw:
        return pw
    pw = getpass.getpass("password: ")
    if confirm and getpass.getpass("again:    ") != pw:
        raise WAVyGravyError("passwords do not match")
    if not pw:
        raise WAVyGravyError("empty password")
    return pw

def parse_band(s):
    try:
        lo, hi = (float(v) for v in s.split("-"))
        assert 0 < lo < hi < WORK_FS / 2
        return lo, hi
    except Exception:
        raise WAVyGravyError("--band must look like 300-2700")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="wavygravy", description="text-in-audio steganography")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--band", default="300-2700", help="carrier band in Hz (must match on decode)")
        p.add_argument("-v", "--verbose", action="store_true")

    e = sub.add_parser("encode", help="encode <cover.wav> <message|file> <output.wav>")
    e.add_argument("cover"); e.add_argument("message"); e.add_argument("output")
    e.add_argument("--redundancy", choices=REDUNDANCY, default="medium")
    e.add_argument("--strength", type=float, default=DEFAULT_STRENGTH_DB,
                   help="dB below the cover (higher = quieter + less robust)")
    e.add_argument("--simulate", choices=["telephone"], help="verify decodability through a simulated channel")
    common(e)

    d = sub.add_parser("decode", help="decode <input.wav> [output.txt]")
    d.add_argument("input"); d.add_argument("output", nargs="?")
    d.add_argument("--max-freq-offset", type=float, default=30.0, help="Hz, SSB mistuning search range")
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
                print(f"  {name:<6} ({L:>4} samples/bit, {WORK_FS / L:5.1f} bit/s raw): "
                      f"{max(capacity_bytes(n8, L), 0)} bytes")
            return 0

        band = parse_band(a.band)
        if a.cmd == "encode":
            fs, cover = read_audio(a.cover)
            msg = open(a.message, "rb").read() if os.path.isfile(a.message) else a.message.encode("utf-8")
            pw = get_password(confirm=sys.stdin.isatty() and "WAVYGRAVY_PASSWORD" not in os.environ)
            out = encode_audio(cover, fs, msg, pw, a.redundancy, a.strength, band)
            write_audio(a.output, fs, out)
            print(f"encoded {len(msg)} bytes → {a.output}", file=sys.stderr)
            if a.simulate:
                fs2, back = read_audio(a.output)
                y, fy = simulate_telephone(back, fs2)
                try:
                    ok = decode_audio(y, fy, pw, band) == msg
                except WAVyGravyError:
                    ok = False
                print(f"simulate {a.simulate}: {'OK' if ok else 'FAILED'}", file=sys.stderr)
                return 0 if ok else 3
            return 0

        fs, x = read_audio(a.input)
        msg = decode_audio(x, fs, get_password(), band, a.max_freq_offset, a.verbose)
        if a.output:
            open(a.output, "wb").write(msg)
        else:
            sys.stdout.buffer.write(msg)
        return 0
    except WAVyGravyError as ex:
        print(f"wavygravy: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

# ── design notes ─────────────────────────────────────────────────────────────
# • The carrier/sync/interleaver key depends on the password alone (the decoder
#   needs it before it can read the salt); the AEAD key also mixes in an 8-byte
#   random salt, so two encodes of the same message differ.
# • Watermark amplitude = (strength) × (local RMS of the band-limited cover).
#   The decoder divides the received audio by its own local RMS (an AGC), so
#   the result is invariant to gain changes and equalises cover interference.
# • Per-bit I/Q correlation makes the decoder tolerant of polarity inversion and
#   arbitrary constant phase rotation; the Costas-style loop follows residual
#   frequency error. The coarse grid search handles SSB mistuning.
# • Not (yet) handled: joining mid-stream (the file must contain the start of
#   the stream), repeated frames / soft-combining, stereo-aware embedding,
#   non-8 kHz-compatible covers (< 8 kHz sample rate).