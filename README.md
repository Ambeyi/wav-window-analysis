# WAV Window Analysis

A Python signal-processing toolkit that reads PCM WAV recordings, normalises each channel and exports sliding-window features. It supports exploratory audio inspection and preparation of numerical features for subsequent analysis.

## Features

- Reads uncompressed 8-bit unsigned, 16-bit signed and 32-bit signed integer PCM WAV files.
- Uses each file's actual sample rate to convert window and hop durations into samples.
- Processes channels independently, preserving stereo signals without downmix cancellation.
- Applies per-channel peak normalisation and handles silent channels without division by zero.
- Exports input RMS, normalised RMS, normalised peak amplitude and zero-crossing rate for each complete window.
- Records sample rate, frame count, channels, duration and processing settings in JSON.
- Includes a reproducible synthetic demonstration and five automated tests.

## Quick start

Requires Python 3.10+ and NumPy. Run commands from the repository directory:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python analyze.py --demo
```

Analyse a WAV file with one-second windows and a half-second hop:

```sh
python analyze.py --wav /path/to/recording.wav --window 1 --hop 0.5 --output output
```

The output directory contains `features.csv` and `audit.json`. Existing files with those names are overwritten. Use a different output directory to retain the bundled demonstration results.

## Demonstration results

The bundled outputs use a generated three-second signal at 8,000 Hz: a 440 Hz sine wave with amplitude 0.5 for two seconds, followed by one second of silence. No personal recordings are included.

| Window in seconds | Input RMS | Normalised RMS |
|---|---:|---:|
| 0.0–1.0 | 0.353553 | 0.707107 |
| 0.5–1.5 | 0.353553 | 0.707107 |
| 1.0–2.0 | 0.353553 | 0.707107 |
| 1.5–2.5 | 0.250000 | 0.500000 |
| 2.0–3.0 | 0.000000 | 0.000000 |

RMS measures signal magnitude. The half-tone, half-silence window has lower RMS; the final window is silent. Input values are fractions of PCM full scale, not calibrated physical measurements.

## Method and validation

For each channel, normalisation divides the signal by its full-recording absolute peak; an entirely silent channel remains zero. RMS is the square root of the mean squared samples. Zero-crossing rate is the fraction of adjacent samples whose sign bits differ, with zero treated as nonnegative. Window end times are exclusive. Durations are rounded to the nearest sample using Python's rounding rule; the JSON records both requested durations and resulting sample counts. Only complete windows are exported.

Five passing tests cover analytical sine-wave RMS and silence, stereo separation and trailing samples, PCM decoding for all three supported widths, invalid inputs and known sign changes. The synthetic demo also runs end to end through the command-line interface.

## Project scope

The project draws on the WAV preprocessing and rolling-window work in Derrick Achanga's Posit Cloud `question2` project. This Python implementation, tests and documentation were prepared with AI assistance. The repository contains signal-feature extraction and synthetic results. It does not provide a trained classifier, movement labels, clinical interpretation or measured prediction accuracy.

Full-recording normalisation is an offline operation and uses the complete recording. Predictive workflows need their own split strategy and preprocessing policy to avoid information leakage. Recordings shorter than one window are rejected; incomplete trailing windows are omitted. Compressed audio, 24-bit PCM and floating-point WAV are outside the supported formats. No resampling or filtering is performed. Users should only process recordings they are authorised to access.
