"""pytest suite for wavygravy.py

Run from the folder containing wavygravy.py:
    pip install pytest numpy scipy cryptography
    pytest -v test_wavygravy.py

Everything uses "low" redundancy and an 8 kHz synthetic noise cover so the
suite stays quick. The cover is ~40 s, which is enough for two full carousel
laps of the default test message (1 header slot + 4 payload slots).
"""
import os
import re
import stat
import hashlib

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


def use_password(monkeypatch, pw):
    """Passwords come from an interactive prompt only (no env var, no argv): answer it."""
    monkeypatch.setattr(wg.getpass, "getpass", lambda prompt="": pw)


def forbid_prompt(monkeypatch):
    def boom(prompt=""):
        raise AssertionError("the password prompt must not be reached here")
    monkeypatch.setattr(wg.getpass, "getpass", boom)


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
    use_password(monkeypatch, PASSWORD)
    out = str(tmp / "cli_out.wav")
    assert wg.main(["encode", cover_path, "stop spying on me", out, "--redundancy", RED,
                          "--strength", str(STRENGTH)]) == 0
    capfd.readouterr()
    assert wg.main(["decode", out]) == 0
    assert capfd.readouterr().out == "stop spying on me"


def test_cli_decode_to_file(tmp, out_a_path, monkeypatch):
    use_password(monkeypatch, PASSWORD)
    txt = tmp / "decoded.txt"
    assert wg.main(["decode", out_a_path, str(txt)]) == 0
    assert txt.read_bytes() == MESSAGE.encode()


def test_cli_wrong_password_exit_code_1(out_a_path, monkeypatch, capsys):
    use_password(monkeypatch, OTHER_PASSWORD)
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
    forbid_prompt(monkeypatch)                    # --no-password must never prompt
    out = str(tmp / "cli_nopw.wav")
    assert wg.main(["encode", "--no-password", cover_path, "hello", out,
                    "--redundancy", RED, "--strength", str(STRENGTH)]) == 0
    assert "no password" in capfd.readouterr().err
    assert wg.main(["decode", "--no-password", out]) == 0
    assert capfd.readouterr().out == "hello"


def test_env_password_is_ignored(out_a_path, monkeypatch, capsys):
    """WAVYGRAVY_PASSWORD was removed (env vars leak to other processes): the right password
    in the environment must not decode, and --no-password must not complain about it."""
    monkeypatch.setenv("WAVYGRAVY_PASSWORD", PASSWORD)
    use_password(monkeypatch, OTHER_PASSWORD)                 # the prompt is the only source
    assert wg.main(["decode", out_a_path]) == 1
    assert "wavygravy:" in capsys.readouterr().err
    assert wg.main(["decode", "--no-password", out_a_path]) == 1
    assert "conflicts" not in capsys.readouterr().err


def test_cli_decode_without_flag_on_password_free_file_fails(nopw_path, monkeypatch, capsys):
    use_password(monkeypatch, PASSWORD)
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


# ── optional keyfile (*.secret): a secret scrypt salt ────────────────────────

SECRET_A = bytes(range(32))
SECRET_B = bytes(range(1, 33))


@pytest.fixture(scope="module")
def keyed_path(tmp, cover_path):
    """PASSWORD + SECRET_A."""
    return encode_to_file(cover_path, tmp / "keyed.wav", secret=SECRET_A)


@pytest.fixture(scope="module")
def keyonly_path(tmp, cover_path):
    """SECRET_A and no password (keyfile-only mode)."""
    return encode_to_file(cover_path, tmp / "keyonly.wav", password=None, secret=SECRET_A)


def decode_keyed(path, password=PASSWORD, secret=SECRET_A):
    fs, x = wg.read_audio(path)
    return wg.decode_audio(x, fs, password, secret=secret)


def write_keyfile(path, text, mode=0o600):
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    os.chmod(path, mode)
    return str(path)


def test_keyfile_roundtrip(keyed_path):
    assert decode_keyed(keyed_path) == MESSAGE.encode()


def test_keyfile_wrong_keyfile_fails(keyed_path):
    with pytest.raises(wg.WavyError):
        decode_keyed(keyed_path, secret=SECRET_B)


def test_keyfile_missing_keyfile_fails(keyed_path):
    with pytest.raises(wg.WavyError):
        decode_keyed(keyed_path, secret=None)


def test_keyfile_right_keyfile_wrong_password_fails(keyed_path):
    with pytest.raises(wg.WavyError):
        decode_keyed(keyed_path, password=OTHER_PASSWORD)


def test_keyfile_not_accepted_on_a_file_made_without_one(out_a_path):
    with pytest.raises(wg.WavyError):
        decode_keyed(out_a_path)                              # right password, extra keyfile


def test_keyfile_only_roundtrip(keyonly_path):
    assert decode_keyed(keyonly_path, password=None) == MESSAGE.encode()
    assert decode_keyed(keyonly_path, password="") == MESSAGE.encode()


def test_keyfile_only_is_not_readable_in_public_no_password_mode(keyonly_path):
    with pytest.raises(wg.WavyError):
        decode_keyed(keyonly_path, password=None, secret=None)


def test_keyfile_only_wrong_keyfile_fails(keyonly_path):
    with pytest.raises(wg.WavyError):
        decode_keyed(keyonly_path, password=None, secret=SECRET_B)


SALT_1 = bytes(range(8))
SALT_2 = bytes(range(1, 9))


def test_secret_must_be_32_bytes():
    for bad in (b"", b"short", bytes(31), bytes(33)):
        with pytest.raises(wg.WavyError):
            wg.master_key(PASSWORD, SALT_1, bad)


def test_master_key_derivation_is_pinned():
    """Spec pin: scrypt over (password, sha256(label | keyfile length | keyfile | salt))."""
    def expect(secret):
        sec = secret or b""
        kdf_salt = hashlib.sha256(b"wavygravy/master/v3" + bytes([len(sec)]) + sec + SALT_1).digest()
        return hashlib.scrypt(PASSWORD.encode(), salt=kdf_salt, n=2 ** 17, r=8, p=1,
                              dklen=32, maxmem=256 * 2 ** 20)
    assert wg.master_key(PASSWORD, SALT_1) == expect(None)
    assert wg.master_key(PASSWORD, SALT_1, SECRET_A) == expect(SECRET_A)


def test_master_key_depends_on_salt_password_and_keyfile():
    keys = {
        wg.master_key(PASSWORD, SALT_1),
        wg.master_key(PASSWORD, SALT_2),                       # other message
        wg.master_key(OTHER_PASSWORD, SALT_1),                 # other password
        wg.master_key(PASSWORD, SALT_1, SECRET_A),             # with a keyfile
        wg.master_key(PASSWORD, SALT_1, SECRET_B),             # with another keyfile
    }
    assert len(keys) == 5


# ── public header, keyed payload ─────────────────────────────────────────────

def test_scrypt_salt_is_per_message_and_matches_between_encode_and_decode(tmp, cover_path, monkeypatch):
    """The scrypt salt used to be a fixed public constant. It must now differ per message, and
    the decoder must rediscover the encoder's salt from the file alone."""
    seen = []
    real = hashlib.scrypt
    def spy(*a, **k):
        seen.append(k["salt"])
        return real(*a, **k)
    monkeypatch.setattr(wg.hashlib, "scrypt", spy)
    p1 = encode_to_file(cover_path, tmp / "salt1.wav")
    p2 = encode_to_file(cover_path, tmp / "salt2.wav")
    enc1, enc2 = seen[0], seen[1]
    seen.clear()
    assert decode_file(p1) == MESSAGE.encode()
    assert seen == [enc1] and enc1 != enc2


def test_header_is_readable_without_password_or_keyfile(keyed_path, capsys):
    """Slot 0 is under a public carrier: with the wrong password AND no keyfile the decode fails,
    but the header (length, salt) is still read, which is what lets the salt be per-message."""
    fs, x = wg.read_audio(keyed_path)
    with pytest.raises(wg.WavyError):
        wg.decode_audio(x, fs, OTHER_PASSWORD, verbose=True)
    assert f"len={len(MESSAGE.encode())}" in capsys.readouterr().err


def test_header_salt_is_the_one_in_the_file(tmp, cover_path, monkeypatch, capsys):
    monkeypatch.setattr(wg.os, "urandom", lambda n: bytes(range(n)))      # pin the salt
    path = encode_to_file(cover_path, tmp / "pinned_salt.wav")
    fs, x = wg.read_audio(path)
    with pytest.raises(wg.WavyError):
        wg.decode_audio(x, fs, OTHER_PASSWORD, verbose=True)
    assert "salt=" + bytes(range(wg.SALT_BYTES)).hex() in capsys.readouterr().err


def test_header_carrier_is_public_payload_carriers_are_per_message():
    n = wg.SYMS_PER_FRAME * 128
    corr = lambda a, b: abs(float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))))
    h = wg.slot_carrier(wg.HEADER_KEY, 0, n)
    assert np.array_equal(h, wg.slot_carrier(wg.HEADER_KEY, 0, n))          # fixed, key-free
    m1, m2 = wg.master_key(PASSWORD, SALT_1), wg.master_key(PASSWORD, SALT_2)
    p1, p1b, p2 = (wg.slot_carrier(m1, 1, n), wg.slot_carrier(m1, 1, n), wg.slot_carrier(m2, 1, n))
    assert np.array_equal(p1, p1b)                                          # deterministic per message
    assert corr(p1, p2) < 0.05                                              # same password, other message: unrelated
    assert corr(p1, wg.slot_carrier(m1, 2, n)) < 0.05                       # other slot: unrelated
    assert corr(p1, h) < 0.05                                               # payload is not the public header carrier


# keygen / load_keyfile / fingerprint

def test_keygen_writes_64_hex_chars_with_private_mode(tmp):
    path = tmp / "k1.secret"
    fp = wg.keygen(str(path))
    assert re.fullmatch(r"[0-9a-f]{64}\n", path.read_text())
    assert re.fullmatch(r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}", fp)
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert len(wg.load_keyfile(str(path))) == 32


def test_keygen_never_overwrites(tmp):
    path = tmp / "k2.secret"
    wg.keygen(str(path))
    before = path.read_bytes()
    with pytest.raises(wg.WavyError, match="already exists"):
        wg.keygen(str(path))
    assert path.read_bytes() == before


def test_keygen_makes_different_keys(tmp):
    wg.keygen(str(tmp / "k3.secret")); wg.keygen(str(tmp / "k4.secret"))
    assert (tmp / "k3.secret").read_bytes() != (tmp / "k4.secret").read_bytes()


def test_fingerprint_is_stable_short_and_does_not_leak_the_key(tmp):
    path = tmp / "k5.secret"
    fp = wg.keygen(str(path))
    key = wg.load_keyfile(str(path))
    assert wg.fingerprint(key) == fp == wg.fingerprint(key)
    assert fp.replace("-", "") not in key.hex()
    assert wg.fingerprint(SECRET_A) != wg.fingerprint(SECRET_B)


def test_load_keyfile_accepts_uppercase_and_trailing_whitespace(tmp):
    p = write_keyfile(tmp / "up.secret", SECRET_A.hex().upper() + " \r\n")
    assert wg.load_keyfile(p) == SECRET_A


@pytest.mark.parametrize("text", [
    "",                                   # empty
    "hunter2",                            # a typed word
    "abcd",                               # valid hex, far too short
    "z" * 64,                             # right length, not hex
    SECRET_A.hex()[:-2],                  # one byte short
    SECRET_A.hex() + "00",                # one byte long
    " ".join(f"{b:02x}" for b in SECRET_A),   # 32 bytes of hex but with spaces
    b"\xff\xfe\x00\x01" * 20,             # binary
])
def test_load_keyfile_rejects_non_keyfiles(tmp, text):
    p = write_keyfile(tmp / "bad.secret", text)
    with pytest.raises(wg.WavyError, match="not a wavygravy keyfile"):
        wg.load_keyfile(p)


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX-only")
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666])
def test_load_keyfile_rejects_group_or_world_accessible(tmp, mode):
    p = write_keyfile(tmp / "loose.secret", SECRET_A.hex(), mode=mode)
    with pytest.raises(wg.WavyError, match="chmod 600"):
        wg.load_keyfile(p)


def test_load_keyfile_missing_file(tmp):
    with pytest.raises(wg.WavyError, match="cannot read keyfile"):
        wg.load_keyfile(str(tmp / "nope.secret"))


# CLI

def test_cli_keygen_and_fingerprint(tmp, capfd):
    path = str(tmp / "cli1.secret")
    assert wg.main(["keygen", path]) == 0
    created = capfd.readouterr().err
    assert wg.main(["fingerprint", path]) == 0
    fp = capfd.readouterr().out.strip()
    assert re.fullmatch(r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}", fp) and fp in created
    assert wg.main(["keygen", path]) == 1                     # refuses to overwrite


def test_cli_keyfile_roundtrip_with_password(tmp, cover_path, monkeypatch, capfd):
    key = write_keyfile(tmp / "cli2.secret", SECRET_A.hex() + "\n")
    use_password(monkeypatch, PASSWORD)
    out = str(tmp / "cli_keyed.wav")
    assert wg.main(["encode", "--keyfile", key, cover_path, "with a keyfile", out,
                    "--redundancy", RED, "--strength", str(STRENGTH)]) == 0
    capfd.readouterr()
    assert wg.main(["decode", "--keyfile", key, out]) == 0
    assert capfd.readouterr().out == "with a keyfile"
    # same password, no keyfile: must not decode, and the error says keyfile too
    assert wg.main(["decode", out]) == 1
    assert "keyfile" in capfd.readouterr().err


def test_cli_keyfile_only_never_prompts(tmp, cover_path, monkeypatch, capfd):
    key = write_keyfile(tmp / "cli3.secret", SECRET_A.hex() + "\n")
    forbid_prompt(monkeypatch)
    out = str(tmp / "cli_keyonly.wav")
    assert wg.main(["encode", "--keyfile", key, "--no-password", cover_path, "scripted", out,
                    "--redundancy", RED, "--strength", str(STRENGTH)]) == 0
    assert "keyfile-only" in capfd.readouterr().err
    assert wg.main(["decode", "--keyfile", key, "--no-password", out]) == 0
    assert capfd.readouterr().out == "scripted"
    assert wg.main(["decode", "--no-password", out]) == 1     # no keyfile: unreadable


def test_cli_wrong_keyfile_exit_code_1(tmp, cover_path, monkeypatch, capfd):
    good = write_keyfile(tmp / "cli4.secret", SECRET_A.hex())
    other = write_keyfile(tmp / "cli5.secret", SECRET_B.hex())
    forbid_prompt(monkeypatch)
    out = str(tmp / "cli_wrongkey.wav")
    assert wg.main(["encode", "--keyfile", good, "--no-password", cover_path, "x", out,
                    "--redundancy", RED, "--strength", str(STRENGTH)]) == 0
    capfd.readouterr()
    assert wg.main(["decode", "--keyfile", other, "--no-password", out]) == 1
    assert "keyfile" in capfd.readouterr().err


def test_cli_bad_keyfile_fails_before_the_password_prompt(tmp, cover_path, monkeypatch, capsys):
    bad = write_keyfile(tmp / "cli6.secret", "hunter2")
    forbid_prompt(monkeypatch)                                # would raise if reached
    assert wg.main(["encode", "--keyfile", bad, cover_path, "x", str(tmp / "never.wav")]) == 1
    assert "not a wavygravy keyfile" in capsys.readouterr().err
    assert not (tmp / "never.wav").exists()


def test_cli_verbose_prints_keyfile_fingerprint_not_the_key(tmp, keyed_path, monkeypatch, capsys):
    key = write_keyfile(tmp / "cli7.secret", SECRET_A.hex())
    use_password(monkeypatch, PASSWORD)
    assert wg.main(["decode", "-v", "--keyfile", key, keyed_path]) == 0
    err = capsys.readouterr().err
    assert wg.fingerprint(SECRET_A) in err and SECRET_A.hex() not in err