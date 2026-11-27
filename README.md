# `wavygravy`

## Disclaimer 

> **For lawful use only:** In the US that includes education, research, and testing, with audio you own or are authorized to modify. To the fullest extent permitted by applicable law: this software is provided “as is” without warranty of any kind, and its authors/contributors disclaim liability for damages or claims arising from its use. This notice is not legal advice and does not guarantee that any particular use is lawful; consult a qualified attorney about your circumstances. You are responsible for ensuring that your use complies with applicable national and local laws and regulations.

> **Radio transmissions:** Do not transmit output from this software unless you are authorized to use the relevant service, frequency, and equipment, and comply with applicable Federal Communications Commission (FCC) rules. Amateur-radio operators in the US should review 47 C.F.R. Part 97, including § 97.113, which prohibits messages encoded for the purpose of obscuring their meaning, except as otherwise permitted by the rule. This software is not represented as compliant with amateur-radio or other radio-service requirements.

> **Audio and content:** You are responsible for having the necessary rights and permissions to modify, reproduce, or transmit any audio or other content you use. Do not use the software to violate another person’s rights, facilitate unlawful conduct, interfere with communications, or evade lawful restrictions.

## Summary

Hide a short UTF-8 text message inside an audio file (WAV or anything ffmpeg can read) using self-locating chirp "ping" beacons and a repeating carousel of keyed-noise-coded chunks. Decoding can begin anywhere in the stream — pings are found by a global matched-filter scan, and each frame re-estimates its own frequency offset and phase from its ping pair, so no drift tracking is needed.

Requires `numpy`, `scipy`, and `cryptography`. `ffmpeg` is used only for non-WAV inputs/outputs.

## Use

```sh
# encode (password via env var or interactive prompt)
WAVYGRAVY_PASSWORD=secret wavygravy.py encode cover.wav "stop spying on me" out.wav

# encode from a file instead of a literal string
WAVYGRAVY_PASSWORD=secret wavygravy.py encode cover.wav message.txt out.wav

# decode (prints to stdout, or writes to a file if given)
WAVYGRAVY_PASSWORD=secret wavygravy.py decode out.wav
WAVYGRAVY_PASSWORD=secret wavygravy.py decode out.wav message.txt

# see how much fits in a cover at each redundancy
wavygravy.py capacity cover.wav
```

## Flags

### `encode cover.wav <message|file> output.wav`

| Flag | Default | Meaning |
|------|---------|---------|
| `--redundancy {low,medium,high}` | `medium` | Samples per data bit (128 / 256 / 512). Higher = more robust, longer frames, less capacity. |
| `--strength DB` | `18.0` | How far below the cover level the data rides (dB). Higher = quieter but less robust; pings ride 6 dB above this. |
| `--band LO-HI` | `300-2700` | Carrier band in Hz. **Must match on decode.** |
| `-v`, `--verbose` | off | Verbose logging. |

### `decode input.wav [output.txt]`

| Flag | Default | Meaning |
|------|---------|---------|
| `--band LO-HI` | `300-2700` | Carrier band in Hz. Must match the encoder. |
| `-v`, `--verbose` | off | Log ping count, frame slot IDs, header details. |

If `output.txt` is omitted, the decoded message is written to stdout.

### `capacity cover.wav`

Prints cover duration and the maximum message size (bytes) that fits at each redundancy level.

## Notes

- **Password** — read from `WAVYGRAVY_PASSWORD` if set, otherwise prompted. It drives both the ChaCha20-Poly1305 message key and the keyed noise carriers; wrong password = no message.
- **Band mismatch** — decode fails silently (reports "no message found"). Keep `--band` consistent between encode and decode.
- **Redundancy auto-try** — the decoder tries `medium`, `high`, then `low` automatically, so you usually don't need to pass `--redundancy` on decode (it doesn't take one).
- **Carousel** — a lap is one pass of header + all payload slots; repeated laps let the decoder sum soft values across laps and self-heal gaps. The more audio you have relative to the message, the more robust it is.
- **Max message** — `16 KiB` hard cap (`MAX_MESSAGE_BYTES`); this is a short-text tool, not a file transfer format.
- **Cover requirements** — must be at least 8 kHz sample rate and long enough to fit the carousel; `capacity` tells you the limit.
- **Trade-off** — chirp pings are visibly recognizable in a spectrogram. This is a deliberate robustness-vs-stealth dial for this draft.
<!-- Add per-install local salt?? -->
