"""Whistle speech-to-text: WAV reading, 30-second chunking and greedy decoding in JAX.

Usage:
    dana transcribe clip.wav
    dana transcribe clip.wav --language de --checkpoint checkpoints/whistle.safetensors
    dana transcribe clip.wav --keywords "Speech Dispatcher,Xiaoyu"
"""

import json
import math
import os
import struct
import sys

import numpy as np

from .model import LANGUAGES

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
DECODE_BUCKET = 64
DEFAULT_BOOST = 3.0  # logit bonus for keyword tokens
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "whistle.safetensors")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer", "whistle.model")

# (format tag, bytes per sample) -> numpy dtype; tag 1 is integer PCM, tag 3 is IEEE float
_WAV_TYPES = {(1, 1): "u1", (1, 2): "<i2", (1, 3): None, (1, 4): "<i4", (3, 4): "<f4", (3, 8): "<f8"}


def read_wav(path):
    """Mono float32 samples in [-1, 1] at 16 kHz from an integer-PCM or float WAV file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    chunks, at = {}, 12
    while at + 8 <= len(data):
        size = struct.unpack_from("<I", data, at + 4)[0]
        chunks.setdefault(data[at:at + 4], data[at + 8:at + 8 + size])
        at += 8 + size + (size & 1)
    if b"fmt " not in chunks or b"data" not in chunks:
        raise ValueError(f"{path} is not a WAV file")
    tag, channels, rate, _, align, _ = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: the real tag starts the sub-format GUID
        tag = struct.unpack_from("<H", chunks[b"fmt "], 24)[0]
    width = align // channels
    if (tag, width) not in _WAV_TYPES:
        raise ValueError(f"{path} is not integer PCM or float WAV")

    raw = chunks[b"data"]
    raw = raw[: len(raw) - len(raw) % align]
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        values = (b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16) << 8 >> 8
    else:
        values = np.frombuffer(raw, _WAV_TYPES[tag, width])
    offset, scale = (0, 1.0) if tag == 3 else (128, 128.0) if width == 1 else (0, float(1 << (8 * width - 1)))
    samples = (values.reshape(-1, channels).mean(axis=1, dtype=np.float64) - offset) / scale
    if rate != SAMPLE_RATE and len(samples):
        from scipy.signal import resample_poly
        g = math.gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // g, rate // g)
    return samples.astype(np.float32)


class WhistleTokenizer:
    """The SentencePiece text vocabulary Whistle was trained on; language tags follow it."""

    def __init__(self, cfg, model_path=TOKENIZER_PATH):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        if self.sp.get_piece_size() != cfg.text:
            raise ValueError(f"{model_path} has {self.sp.get_piece_size()} pieces; the checkpoint expects {cfg.text}")
        self.cfg = cfg
        # Which ids begin a word: "▁" pieces; language tags (past the text vocabulary) do not.
        self.word_start = np.zeros(cfg.vocab, bool)
        self.word_start[:cfg.text] = [self.sp.id_to_piece(i).startswith("▁") for i in range(cfg.text)]

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode([i for i in ids if i != self.cfg.pad and i < self.cfg.text])


def split_keywords(text):
    """Comma-separated words or phrases, as `--keywords` takes them, to a list."""
    return [k.strip() for k in (text or "").split(",") if k.strip()]


class KeywordBias:
    """Favours the token paths of chosen words and phrases during greedy decoding.

    While the text so far ends with the first m tokens of a keyword (m = 0 included), that
    keyword's next token gets `boost` added to its logit, so the model writes the keyword
    whenever it is within `boost` of its own first choice. A keyword only starts where the
    model itself starts a word, so it replaces words, never punctuation or the end.

    The tokenizer marks a word start with "▁" except on a transcript's first word, so
    `starts` are the keywords' token paths at the very start and `paths` everywhere else;
    `word_start[t]` says whether token t is one of those "▁" pieces.
    """

    def __init__(self, starts, paths, word_start, eos, boost=DEFAULT_BOOST):
        self.paths = [(tuple(p), True) for p in starts if p] + [(tuple(p), False) for p in paths if p]
        self.word_start, self.eos, self.boost = word_start, eos, boost

    def _open(self, text):
        """(path, m) for each keyword whose first m tokens end `text`, with m < len(path)."""
        for path, at_start in self.paths:
            for m in range(min(len(path) - 1, len(text)) + 1):
                if (len(text) == m) == at_start and tuple(text[len(text) - m:]) == path[:m]:
                    yield path, m

    def pick(self, logits, text):
        """Greedy choice after the keyword bonus; `logits` is the next-token row."""
        plain = int(logits.argmax())
        new_word = plain != self.eos and (not text or bool(self.word_start[plain]))
        boosted = logits.copy()
        for path, m in self._open(text):
            if m or new_word:
                boosted[path[m]] = logits[path[m]] + self.boost
        return int(boosted.argmax())

    def abandoned(self, text, pick):
        """How many trailing tokens of `text` start a keyword that `pick` leaves unfinished (0 if none)."""
        started = [(path, m) for path, m in self._open(text) if m]
        if any(path[m] == pick for path, m in started):
            return 0
        return max((m for _, m in started), default=0)


class Transcriber:
    """Loads a Whistle checkpoint once and transcribes audio of any length."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, dtype="float32"):
        from .model import load_whistle

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Whistle checkpoint not found: {checkpoint}")
        self._setup(*load_whistle(checkpoint, dtype=dtype))

    @classmethod
    def from_params(cls, params, cfg):
        """A Transcriber over parameters already in memory, e.g. just after finetuning."""
        self = cls.__new__(cls)
        self._setup(params, cfg)
        return self

    def _setup(self, params, cfg):
        import jax
        import jax.numpy as jnp
        from .model import Whistle

        self.cfg = cfg
        self.params = jax.tree.map(jnp.asarray, params)
        self.tokenizer = WhistleTokenizer(self.cfg)
        model = Whistle(self.cfg)
        self._encode = jax.jit(lambda p, wav, lengths: model.apply({"params": p}, wav, lengths,
                                                                    method=Whistle.encode))
        self._logits = jax.jit(lambda p, tokens, enc, frames: model.apply({"params": p}, tokens, enc, frames,
                                                                           method=Whistle.logits))

    def warmup(self):
        """Compile the encoder and the first decoder bucket so the first real call is not slow."""
        c = self.cfg
        enc, frames = self._encode(self.params, np.zeros((1, CHUNK_SECONDS * SAMPLE_RATE), np.float32),
                                   np.array([SAMPLE_RATE], np.int32))
        tokens = np.full((1, DECODE_BUCKET), c.pad, np.int32)
        tokens[0, 0] = c.bos
        self._logits(self.params, tokens, enc, frames).block_until_ready()

    def _decode_chunk(self, samples, language, bias=None):
        """Greedy decode of one chunk of at most CHUNK_SECONDS. Returns (text, language).

        With a KeywordBias, a keyword the bonus started but the model then leaves unfinished
        is taken back: decoding resumes from the first boosted token of it, unboosted there.
        """
        c = self.cfg
        wav = np.zeros((1, CHUNK_SECONDS * SAMPLE_RATE), np.float32)
        wav[0, :len(samples)] = samples
        enc, frames = self._encode(self.params, wav, np.array([len(samples)], np.int32))

        tokens = np.full((1, c.max_tokens), c.pad, np.int32)
        tokens[0, 0] = c.bos
        is_tag = np.arange(c.vocab) >= c.text
        pos = 0
        if language is not None:
            tokens[0, 1] = c.tag(language)
            pos = 1
        boosted, unboosted = set(), set()  # token indices the bonus changed / may no longer change
        while pos < c.max_tokens - 1:
            # Position 0 predicts the language tag; afterwards tags are not text and are masked out.
            size = min(c.max_tokens, (pos // DECODE_BUCKET + 1) * DECODE_BUCKET)
            logits = np.array(self._logits(self.params, tokens[:, :size], enc, frames)[0, pos])
            logits[is_tag != (pos == 0)] = -np.inf
            pick = int(logits.argmax())
            if bias is not None and pos > 0:
                text = tokens[0, 2:pos + 1].tolist()
                if pos + 1 not in unboosted:
                    choice = bias.pick(logits, text)
                    if choice != pick:
                        boosted.add(pos + 1)
                        pick = choice
                taken = [i for i in range(pos + 1 - bias.abandoned(text, pick), pos + 1) if i in boosted]
                if taken:
                    tokens[0, taken[0]:] = c.pad
                    boosted = {i for i in boosted if i < taken[0]}
                    unboosted.add(taken[0])
                    pos = taken[0] - 1
                    continue
            if pick == c.eos:
                break
            tokens[0, pos + 1] = pick
            pos += 1
        language = c.languages[tokens[0, 1] - c.text]
        return self.tokenizer.decode(tokens[0, 2:].tolist()), language

    def transcribe(self, audio, language=None, keywords=None, boost=DEFAULT_BOOST):
        """Speech to text. `audio` is a WAV path or 16 kHz mono float samples.

        `language` is one of the checkpoint's languages, or None to detect it from the
        first 30 seconds. `keywords` are words or phrases to favour, written as they
        should appear; `boost` is how strongly. Returns {"text", "language"}.
        """
        if language is not None and language not in self.cfg.languages:
            raise ValueError(f"language must be one of {', '.join(self.cfg.languages)}")
        bias = None
        if keywords:
            encode = self.tokenizer.encode
            bias = KeywordBias([encode(k) for k in keywords], [encode(" " + k) for k in keywords],
                               self.tokenizer.word_start, self.cfg.eos, boost)
        samples = read_wav(audio) if isinstance(audio, (str, os.PathLike)) else np.asarray(audio, np.float32)
        step = CHUNK_SECONDS * SAMPLE_RATE
        texts = []
        for begin in range(0, max(len(samples), 1), step):
            text, language = self._decode_chunk(samples[begin:begin + step], language, bias)
            texts.append(text.strip())
        return {"text": " ".join(t for t in texts if t), "language": language}


def main(args):
    transcriber = Transcriber(args.checkpoint, dtype=args.dtype)
    keywords = split_keywords(args.keywords)
    for path in args.audio:
        result = transcriber.transcribe(path, language=args.language, keywords=keywords, boost=args.boost)
        if args.json:
            print(json.dumps({"audio": path, **result}, ensure_ascii=False))
        else:
            prefix = f"{path}: " if len(args.audio) > 1 else ""
            print(f"{prefix}[{result['language']}] {result['text']}")
    sys.stdout.flush()


__all__ = ["LANGUAGES", "KeywordBias", "Transcriber", "WhistleTokenizer", "read_wav", "split_keywords", "main"]
