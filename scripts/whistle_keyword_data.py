"""Training clips for teaching Whistle a word from a few recordings of it.

Writes three kinds of example, each with random gain, noise, speed and padding:
  - the word alone ("Dana."), from your recordings (`dana record`);
  - the word next to other speech ("Front left. Dana."), so it is learnt in context;
  - the other speech alone, labelled with the base model's own transcript, so the rest
    of what Whistle writes stays as it was.
The last --holdout recordings go only to val.jsonl, so validation hears a take of the
word that training never did. Output: WAVs plus train.jsonl and val.jsonl, ready for
`dana finetune-whistle`.

  python scripts/whistle_keyword_data.py data/whistle/dana/clips.jsonl --out data/whistle/dana_train \
      --context /usr/share/sounds/alsa/[FRS]*.wav /usr/share/sounds/speech-dispatcher/*.wav
"""
import argparse
import json
import os
import sys
import wave
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dana.whistle.transcribe import SAMPLE_RATE, Transcriber, read_wav  # noqa: E402

SPEEDS = (0.9, 0.95, 1.05, 1.1)


def write_wav(path, samples):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def split_at_pauses(samples, longest):
    """Pieces of at most `longest` seconds, cut in the middle of pauses of 0.25 s or more."""
    if len(samples) <= longest * SAMPLE_RATE:
        return [samples]
    block = SAMPLE_RATE // 20
    rms = np.sqrt([np.mean(samples[i:i + block] ** 2) for i in range(0, len(samples) - block + 1, block)])
    quiet = rms < 0.15 * np.percentile(rms, 95)
    cuts, run = [0], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
            continue
        if run >= 5:
            cuts.append((i - run // 2) * block)
        run = 0
    cuts.append(len(samples))
    pieces, start = [], 0
    for a, b in zip(cuts, cuts[1:]):
        if b - start > longest * SAMPLE_RATE:
            if a > start:
                pieces.append(samples[start:a])
            start = a
    pieces.append(samples[start:])
    return [p for p in pieces if SAMPLE_RATE // 2 <= len(p) <= longest * SAMPLE_RATE]


def augment(x, rng):
    """Random speed, gain, leading/trailing silence and white noise."""
    from scipy.signal import resample_poly
    if rng.random() < 0.7:
        speed = Fraction(rng.choice(SPEEDS)).limit_denominator(20)
        x = resample_poly(x, speed.denominator, speed.numerator)
    x = x * 10 ** (rng.uniform(-10, 6) / 20)
    x = np.concatenate([np.zeros(int(rng.uniform(0.05, 0.6) * SAMPLE_RATE)), x,
                        np.zeros(int(rng.uniform(0.05, 0.6) * SAMPLE_RATE))])
    if rng.random() < 0.8:
        level = np.sqrt(np.mean(x ** 2)) * 10 ** (-rng.uniform(10, 40) / 20)
        x = x + rng.normal(0, level, len(x))
    return np.clip(x, -1, 1).astype(np.float32)


def join(first, second, rng):
    gap = np.zeros(int(rng.uniform(0.15, 0.6) * SAMPLE_RATE), np.float32)
    return np.concatenate([first, gap, second])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", help="JSONL of recordings of the word, from `dana record`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", nargs="+", required=True, help="WAVs of other speech (any speaker, any words)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--checkpoint", default="checkpoints/whistle.safetensors", help="Labels the context speech")
    ap.add_argument("--holdout", type=int, default=1, help="Recordings kept for validation only")
    ap.add_argument("--alone", type=int, default=30, help="Word-alone examples per training recording")
    ap.add_argument("--paired", type=int, default=30, help="Word-with-context examples per training recording")
    ap.add_argument("--plain", type=int, default=120, help="Context-only examples")
    ap.add_argument("--longest", type=float, default=3.5, help="Longest context piece, seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    root = os.path.dirname(os.path.abspath(args.clips))
    with open(args.clips, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    words = [(read_wav(os.path.join(root, r["audio"])), r["text"].strip()) for r in rows]
    if len(words) <= args.holdout:
        raise SystemExit(f"need more than --holdout {args.holdout} recordings, found {len(words)}")
    train_words, val_words = words[: len(words) - args.holdout], words[len(words) - args.holdout:]

    base = Transcriber(args.checkpoint)
    context = []
    for path in args.context:
        for piece in split_at_pauses(read_wav(path), args.longest):
            text = base.transcribe(piece, language=args.language)["text"]
            if text:
                context.append((piece, text))
    if not context:
        raise SystemExit("no usable context speech")
    print(f"{len(train_words)} training and {len(val_words)} held-out recordings, {len(context)} context pieces")
    for piece, text in context:
        print(f"    {len(piece) / SAMPLE_RATE:4.1f} s  {text}")

    def pair(word, rng):
        audio, text = word
        piece, label = context[rng.integers(len(context))]
        if rng.random() < 0.5:
            return join(augment(audio, rng), augment(piece, rng), rng), f"{text} {label}"
        return join(augment(piece, rng), augment(audio, rng), rng), f"{label} {text}"

    examples = {"train": [], "val": []}
    for word in train_words:
        examples["train"] += [(augment(word[0], rng), word[1]) for _ in range(args.alone)]
        examples["train"] += [pair(word, rng) for _ in range(args.paired)]
    for _ in range(args.plain):
        piece, label = context[rng.integers(len(context))]
        examples["train"].append((augment(piece, rng), label))
    for word in val_words:
        examples["val"] += [word] + [(augment(word[0], rng), word[1]) for _ in range(4)]
        examples["val"] += [pair(word, rng) for _ in range(4)]
    examples["val"] += [context[i] for i in rng.choice(len(context), min(6, len(context)), replace=False)]

    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    for split, items in examples.items():
        order = rng.permutation(len(items)) if split == "train" else range(len(items))
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for n, i in enumerate(order, 1):
                audio, text = items[i]
                name = f"wav/{split}_{n:04d}.wav"
                write_wav(os.path.join(args.out, name), audio)
                f.write(json.dumps({"audio": name, "text": text, "language": args.language}, ensure_ascii=False) + "\n")
        seconds = sum(len(a) for a, _ in items) / SAMPLE_RATE
        print(f"{split}: {len(items)} clips, {seconds / 60:.1f} min, longest {max(len(a) for a, _ in items) / SAMPLE_RATE:.1f} s")


if __name__ == "__main__":
    main()
