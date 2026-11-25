"""pytest suite for wavygravy.py

Run from the folder containing wavygravy.py:
    pip install pytest numpy scipy cryptography
    pytest -v test_wavygravy.py

Everything uses "low" redundancy and an 8 kHz synthetic noise cover so the
suite stays quick. The cover is ~40 s, which is enough for two full carousel
laps of the default test message (1 header slot + 4 payload slots).
"""
import os

import numpy as np
import pytest
from scipy.io import wavfile

import wavygravy as wg

FS = 8000
MESSAGE = "stop spying on me"
PASSWORD = "correct horse battery"
OTHER_PASSWORD = "tr0ub4dor&3"
RED = "low"
STRENGTH = 12.0     # dB below cover. The 18 dB default is marginal on pure white-noise covers (see last tests)
COVER_SECONDS = 40


# ── helpers ──────────────────────────────────────────────────────────────────

def make_cover_file(path, seconds=COVER_SECONDS, fs=FS, channels=1, seed=0, amp=0.1):
    rng = np.random.default_rng(seed)
    shape = (int(seconds * fs),) if channels == 1 else (int(seconds * fs), channels)
    wg.write_audio(str(path), fs, rng.normal(0, amp, shape))
    return str(path)


def encode_to_file(cover_path, out_path, message=MESSAGE, password=PASSWORD,
                   redundancy=RED, strength_db=STRENGTH, **kw):
    fs, cover = wg.read_audio(cover_path)
    msg = message if isinstance(message, bytes) else message.encode("utf-8")
    out = wg.encode_audio(cover, fs, msg, password, redundancy, strength_db, **kw)
    wg.write_audio(str(out_path), fs, out)
    return str(out_path)


def decode_file(path, password=PASSWORD):
    fs, x = wg.read_audio(path)
    return wg.decode_audio(x, fs, password)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


# ── fixtures (module-scoped: encoding is the slow part) ──────────────────────

@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("wavygravy")


@pytest.fixture(scope="module")
def cover_path(tmp):
    return make_cover_file(tmp / "cover.wav")


@pytest.fixture(scope="module")
def cover(cover_path):
    return wg.read_audio(cover_path)          # (fs, float array)


@pytest.fixture(scope="module")
def out_a_path(tmp, cover_path):
    return encode_to_file(cover_path, tmp / "out_a.wav")


@pytest.fixture(scope="module")
def out_b_path(tmp, cover_path):
    """Second encode: identical inputs to out_a (salt is random, so it differs)."""
    return encode_to_file(cover_path, tmp / "out_b.wav")


# ── 1. encode ────────────────────────────────────────────────────────────────

def test_encode_output_file_exists(out_a_path):
    assert os.path.isfile(out_a_path)


def test_encode_output_size_at_least_cover(cover_path, out_a_path):
    assert os.path.getsize(out_a_path) >= os.path.getsize(cover_path)


# ── 2. cover integrity ───────────────────────────────────────────────────────
# (the audio analogues of "same char length / whitespace preserved")

def test_cover_same_length_rate_and_channels(cover, out_a_path):
    fs_c, cov = cover
    fs_o, out = wg.read_audio(out_a_path)
    assert fs_o == fs_c
    assert out.shape == cov.shape


def test_stereo_cover_keeps_channel_layout(tmp):
    cp = make_cover_file(tmp / "stereo_cover.wav", channels=2)
    op = encode_to_file(cp, tmp / "stereo_out.wav")
    _, cov = wg.read_audio(cp)
    _, out = wg.read_audio(op)
    assert out.shape == cov.shape and out.ndim == 2
    assert decode_file(op) == MESSAGE.encode()


def test_embedding_is_quiet_relative_to_cover(cover, out_a_path):
    _, cov = cover
    _, out = wg.read_audio(out_a_path)
    snr_db = 20 * np.log10(rms(cov) / rms(out - cov))
    assert snr_db > 10, f"watermark too loud: only {snr_db:.1f} dB below cover"


def test_quiet_passages_stay_quiet(tmp):
    """Audio 'whitespace': the watermark follows the cover's envelope, so a
    quiet stretch should get a much smaller perturbation than a loud one."""
    rng = np.random.default_rng(1)
    n = COVER_SECONDS * FS
    amp = np.where((np.arange(n) // (4 * FS)) % 2 == 0, 0.3, 0.02)   # loud/quiet 4 s blocks
    cp = str(tmp / "dynamic_cover.wav")
    wg.write_audio(cp, FS, rng.normal(0, 1, n) * amp)
    op = encode_to_file(cp, tmp / "dynamic_out.wav")
    _, cov = wg.read_audio(cp)
    _, out = wg.read_audio(op)
    resid = out - cov
    loud = (np.arange(n) // (4 * FS)) % 2 == 0
    assert rms(resid[~loud]) < 0.5 * rms(resid[loud])


# ── 3. decode: correct password ──────────────────────────────────────────────

def test_decode_correct_password_matches(out_a_path):
    assert decode_file(out_a_path) == MESSAGE.encode("utf-8")


# ── 4. decode: wrong password ────────────────────────────────────────────────

def test_decode_wrong_password_raises(out_a_path):
    with pytest.raises(wg.WavyError):
        decode_file(out_a_path, OTHER_PASSWORD)


def test_wrong_password_never_yields_original(out_a_path):
    """Looser property: whatever happens, the plaintext must not leak."""
    try:
        got = decode_file(out_a_path, OTHER_PASSWORD)
    except wg.WavyError:
        return
    assert got != MESSAGE.encode() and got is not None


def test_wrong_password_error_mentions_password(out_a_path):
    with pytest.raises(wg.WavyError, match="password|degraded"):
        decode_file(out_a_path, OTHER_PASSWORD)


# ── 5. same inputs, different outputs, both decode ───────────────────────────

def test_same_inputs_give_different_audio(out_a_path, out_b_path):
    _, a = wg.read_audio(out_a_path)
    _, b = wg.read_audio(out_b_path)
    assert not np.array_equal(a, b)


def test_same_inputs_both_decode(out_a_path, out_b_path):
    assert decode_file(out_a_path) == MESSAGE.encode()
    assert decode_file(out_b_path) == MESSAGE.encode()


def test_two_encodes_have_different_salts(cover, monkeypatch):
    fs, cov = cover
    salts = []
    real = os.urandom
    monkeypatch.setattr(wg.os, "urandom", lambda n: salts.append(real(n)) or salts[-1])
    for _ in range(2):
        wg.encode_audio(cov, fs, b"x", PASSWORD, RED, STRENGTH)
    assert salts[0] != salts[1]


# ── 6. password sensitivity ──────────────────────────────────────────────────

def test_different_password_gives_different_audio(cover, monkeypatch):
    """Pin the salt so the *only* difference between the two encodes is the password."""
    fs, cov = cover
    monkeypatch.setattr(wg.os, "urandom", lambda n: b"\x07" * n)
    a = wg.encode_audio(cov, fs, MESSAGE.encode(), PASSWORD, RED, STRENGTH)
    b = wg.encode_audio(cov, fs, MESSAGE.encode(), OTHER_PASSWORD, RED, STRENGTH)
    assert not np.array_equal(a, b)


def test_pinned_salt_same_password_is_deterministic(cover, monkeypatch):
    fs, cov = cover
    monkeypatch.setattr(wg.os, "urandom", lambda n: b"\x07" * n)
    a = wg.encode_audio(cov, fs, MESSAGE.encode(), PASSWORD, RED, STRENGTH)
    b = wg.encode_audio(cov, fs, MESSAGE.encode(), PASSWORD, RED, STRENGTH)
    np.testing.assert_array_equal(a, b)


# ── 7. cover survives round-trip / different-password round-trip ─────────────

def test_wav_roundtrip_keeps_cover_recognisable(cover, out_a_path):
    _, cov = cover
    _, out = wg.read_audio(out_a_path)
    assert np.corrcoef(cov, out)[0, 1] > 0.95


def test_different_password_full_roundtrip(tmp, cover_path):
    op = encode_to_file(cover_path, tmp / "out_other_pw.wav", password=OTHER_PASSWORD)
    assert decode_file(op, OTHER_PASSWORD) == MESSAGE.encode()
    with pytest.raises(wg.WavyError):
        decode_file(op, PASSWORD)


# ── 8. literal message strings ───────────────────────────────────────────────

def test_cli_literal_string_roundtrip(tmp, cover_path, monkeypatch, capfd):
    monkeypatch.setenv("WAVYGRAVY_PASSWORD", PASSWORD)
    out = str(tmp / "cli_out.wav")
    assert wg.main(["encode", cover_path, "stop spying on me", out, "--redundancy", RED,
                          "--strength", str(STRENGTH)]) == 0
    capfd.readouterr()
    assert wg.main(["decode", out]) == 0
    assert capfd.readouterr().out == "stop spying on me"


def test_cli_decode_to_file(tmp, out_a_path, monkeypatch):
    monkeypatch.setenv("WAVYGRAVY_PASSWORD", PASSWORD)
    txt = tmp / "decoded.txt"
    assert wg.main(["decode", out_a_path, str(txt)]) == 0
    assert txt.read_bytes() == MESSAGE.encode()


def test_cli_wrong_password_exit_code_1(out_a_path, monkeypatch, capsys):
    monkeypatch.setenv("WAVYGRAVY_PASSWORD", OTHER_PASSWORD)
    assert wg.main(["decode", out_a_path]) == 1
    assert "wavygravy:" in capsys.readouterr().err


@pytest.mark.parametrize("message", [
    "",                                  # empty
    "x",                                 # one byte
    "héllo wörld — naïve café",          # non-ASCII
    "emoji 🎧🔒 and 日本語",               # multi-byte UTF-8
    "  leading/trailing spaces\n\ttabs  ",   # whitespace preserved
    "A" * 60,                            # multi-slot payload
])
def test_literal_message_roundtrip(tmp, cover_path, message):
    op = encode_to_file(cover_path, tmp / "lit.wav", message=message)
    got = decode_file(op)
    assert got == message.encode("utf-8")
    assert got.decode("utf-8") == message


# ── 9. extra suggestions ─────────────────────────────────────────────────────

def test_two_outputs_same_password_do_not_cancel(cover, out_a_path, out_b_path):
    """'Zeroing': subtract two same-password outputs (or subtract the cover).
    Salt randomises header + payload, so the difference must not vanish, and
    the two residuals (out - cover) must not be identical watermarks."""
    _, cov = cover
    _, a = wg.read_audio(out_a_path)
    _, b = wg.read_audio(out_b_path)
    assert np.max(np.abs(a - b)) > 0
    ra, rb = a - cov, b - cov
    assert rms(ra) > 0 and rms(rb) > 0
    assert np.corrcoef(ra, rb)[0, 1] < 0.99


def test_residual_is_not_all_zero(cover, out_a_path):
    _, cov = cover
    _, out = wg.read_audio(out_a_path)
    assert np.any(out != cov)


def test_cover_too_short_raises(tmp):
    cp = make_cover_file(tmp / "short.wav", seconds=3)
    fs, cov = wg.read_audio(cp)
    with pytest.raises(wg.WavyError, match="frames"):
        wg.encode_audio(cov, fs, MESSAGE.encode(), PASSWORD, RED)


def test_sample_rate_below_work_rate_raises():
    with pytest.raises(wg.WavyError, match="below"):
        wg.encode_audio(np.zeros(4000), 4000, b"hi", PASSWORD, RED)


def test_noise_only_audio_has_no_message(tmp):
    cp = make_cover_file(tmp / "plain.wav", seed=5)
    with pytest.raises(wg.WavyError, match="ping|decoded"):
        decode_file(cp)


def test_custom_band_roundtrip_and_wrong_band_fails(cover):
    fs, cov = cover
    out = wg.encode_audio(cov, fs, MESSAGE.encode(), PASSWORD, RED, STRENGTH, band=(400.0, 2000.0))
    assert wg.decode_audio(out, fs, PASSWORD, band=(400.0, 2000.0)) == MESSAGE.encode()
    # a band that excludes the 350-2650 Hz ping chirps entirely -> no beacons found
    with pytest.raises(wg.WavyError):
        wg.decode_audio(out, fs, PASSWORD, band=(3000.0, 3900.0))


def test_non_8k_cover_resamples_and_decodes(tmp):
    cp = make_cover_file(tmp / "cover44.wav", seconds=COVER_SECONDS, fs=44100)
    op = encode_to_file(cp, tmp / "out44.wav")
    assert wg.read_audio(op)[0] == 44100
    assert decode_file(op) == MESSAGE.encode()


def test_decoder_can_join_mid_stream(out_a_path):
    """Carousel property: drop the first ~3.5 frames, decoder must still lock on."""
    fs, x = wg.read_audio(out_a_path)
    cut = int(3.5 * wg.frame_len_samples(wg.REDUNDANCY[RED]) * fs / wg.WORK_FS)
    assert wg.decode_audio(x[cut:], fs, PASSWORD) == MESSAGE.encode()


def test_survives_mild_added_noise(out_a_path):
    fs, x = wg.read_audio(out_a_path)
    noisy = x + np.random.default_rng(3).normal(0, 0.01, x.shape)
    assert wg.decode_audio(noisy, fs, PASSWORD) == MESSAGE.encode()


def test_survives_dropout_of_one_data_field(out_a_path):
    """Silence the data field (not the ping) of slot 2 in lap 1; lap 2 should heal it."""
    fs, x = wg.read_audio(out_a_path)
    flen = wg.frame_len_samples(wg.REDUNDANCY[RED])
    d0 = 2 * flen + 2 * wg.CHIRP_LEN + 200
    y = x.copy()
    y[d0:d0 + flen - 2 * wg.CHIRP_LEN - 400] = 0
    assert wg.decode_audio(y, fs, PASSWORD) == MESSAGE.encode()


def test_survives_dropout_of_entire_frame(out_a_path):
    """Zero a whole frame (ping + data) of lap 1; later laps must still recover the message."""
    fs, x = wg.read_audio(out_a_path)
    flen = wg.frame_len_samples(wg.REDUNDANCY[RED])
    y = x.copy()
    y[2 * flen:3 * flen] = 0
    assert wg.decode_audio(y, fs, PASSWORD) == MESSAGE.encode()


@pytest.mark.xfail(reason="Known marginal case: at the default 18 dB strength on a pure white-noise "
                          "cover roughly 1 in 5 salts fail to decode even with no channel damage "
                          "(header decodes, payload does not). Fixtures use STRENGTH=12 to avoid it.",
                   strict=False)
def test_default_strength_on_white_noise_cover_always_decodes(monkeypatch):
    cov = np.random.default_rng(0).normal(0, 0.1, COVER_SECONDS * FS)
    failures = []
    for k in range(1, 11):
        monkeypatch.setattr(wg.os, "urandom", lambda n, k=k: bytes([k]) * n)
        out = wg.encode_audio(cov, FS, MESSAGE.encode(), "pw", RED)     # default strength
        try:
            ok = wg.decode_audio(out, FS, "pw") == MESSAGE.encode()
        except wg.WavyError:
            ok = False
        if not ok:
            failures.append(k)
    assert not failures, f"salts that failed to decode: {failures}"


# ── partial recovery: windows cut out of the output ─────────────────────────
# At "low" redundancy a frame is 26016 samples (3.25 s). A decoder window must hold
# one lap-opening (boundary) ping plus every slot after it. A message of <= 4 bytes
# needs 1 header + 2 payload slots = 3 slots per lap, so any half of a 40 s file
# (>= 5 whole frames wherever it is cut) is guaranteed to contain one complete lap.

SMALL_MESSAGE = b"help"                                  # 4 bytes -> 3-slot carousel
HALF_STARTS_S = [0.0, 0.4, 1.7, 3.3, 5.0, 7.9, 10.0, 13.7, 16.1, 19.99]   # cut offsets, seconds


@pytest.fixture(scope="module")
def small_out(tmp, cover_path):
    path = encode_to_file(cover_path, tmp / "small_out.wav", message=SMALL_MESSAGE)
    return wg.read_audio(path)                           # (fs, float array)


@pytest.mark.parametrize("start_s", HALF_STARTS_S)
def test_small_message_recovered_from_half_length_window(small_out, start_s):
    fs, x = small_out
    half = len(x) // 2
    a = int(start_s * fs)
    window = x[a:a + half]
    assert len(window) == half
    assert wg.decode_audio(window, fs, PASSWORD) == SMALL_MESSAGE


def test_small_message_recovered_from_random_half_windows(small_out):
    fs, x = small_out
    half = len(x) // 2
    rng = np.random.default_rng(42)                      # seeded: reproducible "random" cuts
    for a in rng.integers(0, len(x) - half + 1, size=6):
        assert wg.decode_audio(x[a:a + half], fs, PASSWORD) == SMALL_MESSAGE, f"start sample {a}"


def test_both_halves_decode_independently(small_out):
    fs, x = small_out
    half = len(x) // 2
    assert wg.decode_audio(x[:half], fs, PASSWORD) == SMALL_MESSAGE
    assert wg.decode_audio(x[half:], fs, PASSWORD) == SMALL_MESSAGE


def test_large_message_not_recoverable_from_second_half(out_a_path):
    """Limit check: the 17-byte message needs 5 slots per lap, and the second half of the
    40 s file holds a boundary ping with too few frames after it. Must fail cleanly."""
    fs, x = wg.read_audio(out_a_path)
    with pytest.raises(wg.WavyError):
        wg.decode_audio(x[len(x) // 2:], fs, PASSWORD)


def test_ten_second_window_is_too_short_to_decode(small_out):
    """10 s is ~3 frames at low redundancy; cut away from a frame boundary it cannot hold
    a boundary ping plus a full 3-slot lap, so it must fail cleanly (never a wrong message)."""
    fs, x = small_out
    a = int(5.0 * fs)
    with pytest.raises(wg.WavyError):
        wg.decode_audio(x[a:a + 10 * fs], fs, PASSWORD)


# ── password-free (null password) mode ───────────────────────────────────────

@pytest.fixture(scope="module")
def nopw_path(tmp, cover_path):
    fs, cov = wg.read_audio(cover_path)
    out = wg.encode_audio(cov, fs, SMALL_MESSAGE, None, RED, STRENGTH)
    path = str(tmp / "nopw.wav")
    wg.write_audio(path, fs, out)
    return path


@pytest.mark.parametrize("pw", [None, ""])
def test_null_password_roundtrip(nopw_path, pw):
    """None and "" are interchangeable: encoded with None, decodes with either."""
    assert decode_file(nopw_path, pw) == SMALL_MESSAGE


def test_null_password_file_rejects_real_password(nopw_path):
    with pytest.raises(wg.WavyError):
        decode_file(nopw_path, PASSWORD)


def test_password_file_rejects_null_password(out_a_path):
    with pytest.raises(wg.WavyError):
        decode_file(out_a_path, None)
    with pytest.raises(wg.WavyError):
        decode_file(out_a_path, "")


def test_null_password_empty_string_encode_decodes_with_none(tmp, cover_path):
    op = encode_to_file(cover_path, tmp / "nopw_empty.wav", message=SMALL_MESSAGE, password="")
    assert decode_file(op, None) == SMALL_MESSAGE


def test_null_password_still_salted_so_outputs_differ(cover):
    fs, cov = cover
    a = wg.encode_audio(cov, fs, SMALL_MESSAGE, None, RED, STRENGTH)
    b = wg.encode_audio(cov, fs, SMALL_MESSAGE, None, RED, STRENGTH)
    assert not np.array_equal(a, b)
    assert wg.decode_audio(a, fs, None) == SMALL_MESSAGE
    assert wg.decode_audio(b, fs, None) == SMALL_MESSAGE


def test_null_password_works_from_half_window(nopw_path):
    fs, x = wg.read_audio(nopw_path)
    half = len(x) // 2
    assert wg.decode_audio(x[3 * fs:3 * fs + half], fs, None) == SMALL_MESSAGE


def test_cli_no_password_roundtrip(tmp, cover_path, monkeypatch, capfd):
    monkeypatch.delenv("WAVYGRAVY_PASSWORD", raising=False)
    out = str(tmp / "cli_nopw.wav")
    assert wg.main(["encode", "--no-password", cover_path, "hello", out,
                    "--redundancy", RED, "--strength", str(STRENGTH)]) == 0
    assert "no password" in capfd.readouterr().err
    assert wg.main(["decode", "--no-password", out]) == 0
    assert capfd.readouterr().out == "hello"


def test_cli_no_password_conflicts_with_env_password(out_a_path, monkeypatch, capsys):
    monkeypatch.setenv("WAVYGRAVY_PASSWORD", PASSWORD)
    assert wg.main(["decode", "--no-password", out_a_path]) == 1
    assert "conflicts" in capsys.readouterr().err


def test_cli_decode_without_flag_on_password_free_file_fails(nopw_path, monkeypatch, capsys):
    monkeypatch.setenv("WAVYGRAVY_PASSWORD", PASSWORD)
    assert wg.main(["decode", nopw_path]) == 1


# ── FEC unit tests (fast, no audio) ──────────────────────────────────────────

def test_fec_roundtrip_clean():
    data = bytes(range(32))
    soft = 1.0 - 2.0 * wg.fec_encode(data).astype(float)
    assert wg.fec_decode(soft) == data


def test_fec_corrects_scattered_bit_errors():
    data = bytes(range(32))
    soft = 1.0 - 2.0 * wg.fec_encode(data).astype(float)
    soft[[5, 60, 130, 200, 301]] *= -1          # flip a few well-separated coded bits
    assert wg.fec_decode(soft) == data


def test_capacity_bytes_monotonic_and_fits():
    assert wg.capacity_bytes(0, 128) == -1
    prev = -1
    for frames in range(1, 30):
        cap = wg.capacity_bytes(frames, 128)
        assert cap >= prev
        if cap >= 0:
            assert 1 + wg.n_payload_slots(cap) <= frames
            assert 1 + wg.n_payload_slots(cap + 1) > frames
        prev = cap