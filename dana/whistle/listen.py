"""Live speech-to-text from the microphone with Whistle.

Audio comes from `arecord` (or raw audio on stdin) in 100 ms blocks. Each block's
loudness is compared with a threshold calibrated from the first second of room
noise; a stretch of speech followed by `--silence` seconds of quiet is one
utterance, which is transcribed while recording continues in the background.

Usage:
    dana listen                          # default microphone, Ctrl+C to stop
    dana listen --language de --silence 1.0
    dana listen --keywords "Dana,Xiaoyu"     # Dana is favoured by default; --keywords "" for none
    dana listen --tools examples/tools/smart_home.json   # also favour the tools' words
    dana listen --beam-size 1                # greedy: faster, a little less accurate (default: beam 4)
    arecord -f S16_LE -r 16000 -c 1 -t raw | dana listen --stdin
"""

import collections
import itertools
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from .transcribe import CHUNK_SECONDS, DEFAULT_CHECKPOINT, SAMPLE_RATE, Transcriber, split_keywords

# Finetuned to hear "Dana" (see GUIDE_zh.md 4.6); used instead of the base weights when present.
TUNED_CHECKPOINT = os.path.join("checkpoints", "whistle_dana.safetensors")
BLOCK_SECONDS = 0.1
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_SECONDS)
MIN_THRESHOLD = 0.005  # RMS floor so a silent room does not make every click "speech"


def _rms(block):
    return float(np.sqrt(np.mean(block ** 2))) if len(block) else 0.0


def blocks_from(stream):
    """Float32 blocks of BLOCK_SAMPLES from raw 16-bit little-endian mono 16 kHz bytes."""
    while True:
        data = stream.read(BLOCK_SAMPLES * 2)
        if not data:
            return
        yield np.frombuffer(data[: len(data) - len(data) % 2], "<i2").astype(np.float32) / 32768.0


def _in_background(blocks):
    """Keep reading `blocks` on a thread so audio is not dropped while a transcription runs."""
    q = queue.Queue()

    def pump():
        for block in blocks:
            q.put(block)
        q.put(None)

    threading.Thread(target=pump, daemon=True).start()
    while (block := q.get()) is not None:
        yield block


def calibrate(blocks, seconds=1.0):
    """Speech threshold: 3x the median loudness of the next `seconds` of (quiet) audio."""
    levels = [_rms(b) for b in itertools.islice(blocks, round(seconds / BLOCK_SECONDS))]
    return max(3 * float(np.median(levels)) if levels else 0.0, MIN_THRESHOLD)


def utterances(blocks, threshold, silence=0.8, preroll=0.3, min_speech=0.3):
    """Split audio blocks into utterances: speech that ends after `silence` seconds of quiet.

    Keeps `preroll` seconds before the first loud block so word onsets are not clipped, caps
    an utterance at Whistle's 30-second window, and drops ones with under `min_speech`
    seconds of loud audio (coughs, clicks).
    """
    quiet_needed = max(1, round(silence / BLOCK_SECONDS))
    max_blocks = round(CHUNK_SECONDS / BLOCK_SECONDS)
    before = collections.deque(maxlen=max(1, round(preroll / BLOCK_SECONDS)))
    speech, quiet, voiced = [], 0, 0
    for block in blocks:
        loud = _rms(block) > threshold
        if not speech:
            before.append(block)
            if loud:
                speech, quiet, voiced = list(before), 0, 1
                before.clear()
            continue
        speech.append(block)
        voiced += loud
        quiet = 0 if loud else quiet + 1
        if quiet >= quiet_needed or len(speech) >= max_blocks:
            if voiced * BLOCK_SECONDS >= min_speech:
                # Keep only `preroll` seconds of the trailing quiet, matching the lead-in.
                yield np.concatenate(speech[: len(speech) - max(0, quiet - before.maxlen)])
            speech = []
    if speech and voiced * BLOCK_SECONDS >= min_speech:
        yield np.concatenate(speech)


def _arecord(device):
    if not shutil.which("arecord"):
        raise SystemExit("arecord not found (install alsa-utils), or pipe raw 16 kHz S16_LE mono audio "
                         "into `dana listen --stdin`")
    cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1"]
    if device:
        cmd += ["-D", device]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def load_transcriber(args):
    """The Transcriber `--checkpoint` names (the finetuned weights if present, by default), warmed up."""
    checkpoint = args.checkpoint or (TUNED_CHECKPOINT if os.path.exists(TUNED_CHECKPOINT) else DEFAULT_CHECKPOINT)
    print(f"Loading Whistle ({checkpoint})...", file=sys.stderr, flush=True)
    transcriber = Transcriber(checkpoint, dtype=args.dtype)
    if args.language is not None and args.language not in transcriber.cfg.languages:
        raise SystemExit(f"--language must be one of {', '.join(transcriber.cfg.languages)}")
    transcriber.warmup(getattr(args, "beam_size", 4))
    return transcriber


def heard(args, transcriber, keywords=None):
    """Yield (result, audio seconds, transcribe seconds) for each utterance with text in it.

    `keywords` defaults to --keywords; items may be (word, bonus) pairs."""
    keywords = split_keywords(args.keywords) if keywords is None else keywords
    beam_size = getattr(args, "beam_size", 4)
    recorder = None if args.stdin else _arecord(args.device)
    blocks = _in_background(blocks_from(sys.stdin.buffer if args.stdin else recorder.stdout))
    try:
        threshold = args.threshold
        if threshold is None:
            print(f"Measuring room noise for {args.calibrate:g} s, stay quiet...", file=sys.stderr, flush=True)
            threshold = calibrate(blocks, args.calibrate)
        plain = [k for k in keywords if isinstance(k, str)]
        favour = f", keywords {', '.join(plain)}" if plain else ""
        favour += f", beam {beam_size}" if beam_size > 1 else ""
        print(f"Listening (threshold {threshold:.4f}{favour}). Speak; Ctrl+C to stop.", file=sys.stderr, flush=True)
        for clip in utterances(blocks, threshold, silence=args.silence):
            start = time.time()
            result = transcriber.transcribe(clip, language=args.language, keywords=keywords, boost=args.boost,
                                            beam_size=beam_size)
            if result["text"]:
                yield result, len(clip) / SAMPLE_RATE, time.time() - start
    finally:
        if recorder is not None:
            recorder.terminate()


def main(args):
    from ..voice import whistle_keywords

    transcriber = load_transcriber(args)
    keywords = whistle_keywords(args)
    try:
        for result, seconds, spent in heard(args, transcriber, keywords):
            print(f"[{result['language']}] {result['text']}", flush=True)
            print(f"    ({seconds:.1f} s audio, {spent:.1f} s to transcribe)", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
