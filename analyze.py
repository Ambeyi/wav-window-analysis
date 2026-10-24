"""Inspect PCM WAV files and export window-level signal features."""
import argparse
import csv
import json
import math
from pathlib import Path
import wave
import numpy as np


def read_wav(path):
    with wave.open(str(path), 'rb') as wav:
        width, channels, rate, count = wav.getsampwidth(), wav.getnchannels(), wav.getframerate(), wav.getnframes()
        if width not in (1, 2, 4) or wav.getcomptype() != 'NONE':
            raise ValueError('Use uncompressed 8-, 16-, or 32-bit integer PCM WAV')
        raw = wav.readframes(count)
    if not count or channels < 1 or rate < 1:
        raise ValueError('Audio must contain frames with a valid sample rate')
    if len(raw) != count * channels * width:
        raise ValueError('Truncated PCM data')
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(float) - 128) / 128
    else:
        samples = np.frombuffer(raw, dtype='<i2' if width == 2 else '<i4').astype(float) / (2 ** (8 * width - 1))
    # Channels remain separate to avoid cancellation during stereo downmixing.
    return samples.reshape(-1, channels), rate


def features(samples, rate, window_seconds=1.0, hop_seconds=0.5):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or not samples.size or not np.isfinite(samples).all():
        raise ValueError('Require a nonempty finite frames-by-channels array')
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError('Sample rate must be a positive integer')
    if not all(math.isfinite(x) and x > 0 for x in [window_seconds, hop_seconds]):
        raise ValueError('Window and hop must be positive and finite')
    width, hop = round(window_seconds * rate), round(hop_seconds * rate)
    if width < 2 or hop < 1:
        raise ValueError('Window must span at least two samples and hop at least one')
    if len(samples) < width:
        raise ValueError('Audio is shorter than one complete window')
    peaks = np.max(np.abs(samples), axis=0)
    normalized = np.divide(samples, peaks, out=np.zeros_like(samples), where=peaks > 0)
    rows = []
    for channel in range(samples.shape[1]):
        for start in range(0, len(samples) - width + 1, hop):
            segment = normalized[start:start + width, channel]
            raw = samples[start:start + width, channel]
            rows.append({
                'channel': channel + 1, 'start_seconds': start / rate,
                'end_seconds': (start + width) / rate,
                'rms_input': float(np.sqrt(np.mean(raw ** 2))),
                'rms_normalized': float(np.sqrt(np.mean(segment ** 2))),
                'peak_normalized': float(np.max(np.abs(segment))),
                'zero_crossing_rate': float(np.mean(np.signbit(segment[1:]) != np.signbit(segment[:-1])))})
    return rows


def demo():
    rate = 8000
    t = np.arange(3 * rate) / rate
    # Two seconds of a 440 Hz tone followed by one second of silence.
    signal = np.where(t < 2, 0.5 * np.sin(2 * np.pi * 440 * t), 0)
    return signal[:, None], rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--wav', type=Path)
    source.add_argument('--demo', action='store_true')
    parser.add_argument('--window', type=float, default=1.0)
    parser.add_argument('--hop', type=float, default=0.5)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    samples, rate = demo() if args.demo else read_wav(args.wav)
    rows = features(samples, rate, args.window, args.hop)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'features.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {'source': 'synthetic 440 Hz tone and silence' if args.demo else 'user-supplied WAV',
             'sample_rate_hz': rate, 'frames': len(samples), 'channels': samples.shape[1],
             'duration_seconds': len(samples)/rate, 'window_seconds_requested': args.window,
             'hop_seconds_requested': args.hop, 'window_samples': round(args.window*rate),
             'hop_samples': round(args.hop*rate), 'feature_rows': len(rows),
             'tail_policy': 'Only complete windows are exported; trailing incomplete windows are omitted.',
             'normalization': 'Independent full-recording peak normalization per channel; silence remains zero.'}
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
