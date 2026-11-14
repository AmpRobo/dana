"""Record short labelled clips from the microphone, for finetuning Whistle.

Each utterance (speech followed by `--silence` seconds of quiet, as in `dana listen`)
becomes one 16 kHz mono WAV, and a line {"audio", "text", "language"} is appended to
clips.jsonl in the same folder; recording more later continues the numbering.

Usage:
    dana record "Dana." --count 5                  # writes data/whistle/clips/clip_001.wav ...
    dana record "Call Dana tomorrow." --count 3 --out data/whistle/clips
"""

import json
import os
import sys
import wave

import numpy as np

from .listen import _arecord, _in_background, blocks_from, calibrate, utterances
from .transcribe import SAMPLE_RATE


def write_wav(path, samples):
    """16-bit PCM mono WAV at SAMPLE_RATE from float samples in [-1, 1]."""
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def main(args):
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "clips.jsonl")
    taken = {name for name in os.listdir(args.out) if name.startswith("clip_") and name.endswith(".wav")}
    number = 1

    recorder = _arecord(args.device)
    blocks = _in_background(blocks_from(recorder.stdout))
    try:
        threshold = args.threshold
        if threshold is None:
            print(f"Measuring room noise for {args.calibrate:g} s, stay quiet...", flush=True)
            threshold = calibrate(blocks, args.calibrate)
        print(f"Ready (threshold {threshold:.4f}). Pause about a second after each one.", flush=True)
        clips = utterances(blocks, threshold, silence=args.silence)
        for k in range(1, args.count + 1):
            print(f"  ({k}/{args.count}) say: {args.text}", flush=True)
            clip = next(clips, None)
            if clip is None:
                break
            while f"clip_{number:03d}.wav" in taken:
                number += 1
            name = f"clip_{number:03d}.wav"
            taken.add(name)
            write_wav(os.path.join(args.out, name), clip)
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps({"audio": name, "text": args.text, "language": args.language},
                                   ensure_ascii=False) + "\n")
            print(f"        saved {name} ({len(clip) / SAMPLE_RATE:.1f} s)", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.terminate()
    print(f"Clips and {os.path.basename(manifest)} are in {args.out}", file=sys.stderr)
