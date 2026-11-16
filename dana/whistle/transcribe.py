"""Whistle speech-to-text: WAV reading, 30-second chunking and greedy or beam-search decoding in JAX.

Usage:
    dana transcribe clip.wav
    dana transcribe clip.wav --language de --checkpoint checkpoints/whistle.safetensors
    dana transcribe clip.wav --keywords "Speech Dispatcher,Xiaoyu"
    dana transcribe clip.wav --tools examples/tools/smart_home.json   # favour the tools' words
    dana transcribe clip.wav --beam-size 1                            # greedy (default: beam 4)
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

    def __init__(self, starts, paths, word_start, eos, boost=DEFAULT_BOOST, start_boosts=None, path_boosts=None):
        """`start_boosts` / `path_boosts`, when given, are one bonus per entry of `starts` /
        `paths`, so some keywords (say, words taken from tool definitions) can be favoured
        less than others; otherwise every keyword gets `boost`."""
        start_boosts = start_boosts or [boost] * len(starts)
        path_boosts = path_boosts or [boost] * len(paths)
        self.paths = ([(tuple(p), True, b) for p, b in zip(starts, start_boosts) if p]
                      + [(tuple(p), False, b) for p, b in zip(paths, path_boosts) if p])
        self.word_start, self.eos, self.boost = word_start, eos, boost

    def _open(self, text, with_boost=False):
        """(path, m) for each keyword whose first m tokens end `text`, with m < len(path)."""
        for path, at_start, boost in self.paths:
            for m in range(min(len(path) - 1, len(text)) + 1):
                if (len(text) == m) == at_start and tuple(text[len(text) - m:]) == path[:m]:
                    yield (path, m, boost) if with_boost else (path, m)

    def bonuses(self, logits, text):
        """{token: bonus} for the next token after `text`; `logits` is the next-token row.

        A keyword's first token only gets its bonus where the model's own first choice
        starts a new word; a keyword already under way always gets it."""
        plain = int(logits.argmax())
        new_word = plain != self.eos and (not text or bool(self.word_start[plain]))
        found = {}
        for path, m, boost in self._open(text, with_boost=True):
            if m or new_word:
                found[path[m]] = max(found.get(path[m], 0.0), boost)
        return found

    def completes(self, text):
        """Whether `text` ends with a whole keyword (at the very start for start paths)."""
        n = len(text)
        return any(len(path) <= n and tuple(text[n - len(path):]) == path and (n == len(path)) == at_start
                   for path, at_start, _ in self.paths)

    def pick(self, logits, text):
        """Greedy choice after the keyword bonus; `logits` is the next-token row."""
        boosted = logits.copy()
        for token, bonus in self.bonuses(logits, text).items():
            boosted[token] = logits[token] + bonus
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
        self._jnp = jnp
        self.params = jax.tree.map(jnp.asarray, params)
        self.tokenizer = WhistleTokenizer(self.cfg)
        model = Whistle(self.cfg)
        self._encode = jax.jit(lambda p, wav, lengths: model.apply({"params": p}, wav, lengths,
                                                                    method=Whistle.encode))
        self._logits = jax.jit(lambda p, tokens, enc, frames: model.apply({"params": p}, tokens, enc, frames,
                                                                           method=Whistle.logits))

    def warmup(self, beam_size=1):
        """Compile the encoder and the first decoder bucket so the first real call is not slow."""
        c = self.cfg
        enc, frames = self._encode(self.params, np.zeros((1, CHUNK_SECONDS * SAMPLE_RATE), np.float32),
                                   np.array([SAMPLE_RATE], np.int32))
        tokens = np.full((1, DECODE_BUCKET), c.pad, np.int32)
        tokens[0, 0] = c.bos
        self._logits(self.params, tokens, enc, frames).block_until_ready()
        if beam_size > 1:
            jnp = self._jnp
            tokens = np.repeat(tokens, beam_size, axis=0)
            self._logits(self.params, tokens, jnp.repeat(enc, beam_size, axis=0),
                         jnp.repeat(frames, beam_size, axis=0)).block_until_ready()

    def _decode_chunk(self, samples, language, bias=None, beam_size=1):
        """Decode one chunk of at most CHUNK_SECONDS. Returns (text, language).

        Greedy with beam_size 1. With a KeywordBias, a keyword the bonus started but the
        model then leaves unfinished is taken back: decoding resumes from the first boosted
        token of it, unboosted there.
        """
        c = self.cfg
        wav = np.zeros((1, CHUNK_SECONDS * SAMPLE_RATE), np.float32)
        wav[0, :len(samples)] = samples
        enc, frames = self._encode(self.params, wav, np.array([len(samples)], np.int32))
        if beam_size > 1:
            return self._beam_search(enc, frames, language, bias, beam_size)

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

    def _language_tag(self, enc, frames, language):
        """The language tag token: the one given, or the model's first choice at position 0."""
        c = self.cfg
        if language is not None:
            return c.tag(language)
        tokens = np.full((1, DECODE_BUCKET), c.pad, np.int32)
        tokens[0, 0] = c.bos
        logits = np.array(self._logits(self.params, tokens, enc, frames)[0, 0])
        logits[np.arange(c.vocab) < c.text] = -np.inf
        return int(logits.argmax())

    def _beam_search(self, enc, frames, language, bias, beam_size):
        """Beam search over `beam_size` hypotheses, decoded together as one batch.

        Each step every live hypothesis proposes its best next tokens; the `beam_size` best
        continuations by total log-probability (plus keyword bonuses) survive, and ones that
        end go to `finished`. The search stops once no live hypothesis scores above the best
        finished one (scores only fall), and the finished one with the best log-probability
        per token wins, so short ones are not favoured.

        Keyword bonuses count only for keywords a hypothesis completes: while it is part way
        through one, the bonuses it collected are owed, and paid back (subtracted) if it
        turns away before the end, the beam version of greedy decoding's take-back.
        """
        c, jnp, k = self.cfg, self._jnp, beam_size
        tag = self._language_tag(enc, frames, language)
        enc_k, frames_k = jnp.repeat(enc, k, axis=0), jnp.repeat(frames, k, axis=0)
        is_tag = np.arange(c.vocab) >= c.text
        beams = [([], 0.0, 0.0)]  # (text tokens after the tag, score, bonus owed)
        finished = []  # (score, text tokens)
        pos = 1  # index of the last filled position: 0 is BOS, 1 the language tag
        while beams and pos < c.max_tokens - 1:
            size = min(c.max_tokens, (pos // DECODE_BUCKET + 1) * DECODE_BUCKET)
            batch = np.full((k, size), c.pad, np.int32)
            batch[:, 0], batch[:, 1] = c.bos, tag
            for i in range(k):  # rows past the live beams repeat beam 0, keeping the batch shape fixed
                batch[i, 2:pos + 1] = beams[i if i < len(beams) else 0][0]
            logits = np.array(self._logits(self.params, batch, enc_k, frames_k)[:, pos])
            logits[:, is_tag] = -np.inf
            peak = logits.max(axis=1, keepdims=True)
            logp = logits - peak - np.log(np.exp(logits - peak).sum(axis=1, keepdims=True))

            candidates = []
            for i, (text, score, _) in enumerate(beams):
                row = logp[i].copy()
                bonus = bias.bonuses(logits[i], text) if bias is not None else {}
                for token, extra in bonus.items():
                    row[token] += extra
                for token in np.argpartition(-row, k)[:k + 1]:
                    candidates.append((score + float(row[token]), i, int(token), bonus.get(int(token), 0.0)))
            candidates.sort(key=lambda x: -x[0])

            survivors = []
            for total, i, token, extra in candidates:
                text, _, owed = beams[i]
                if token == c.eos:
                    finished.append((total - owed, text))
                    continue
                text = text + [token]
                if bias is not None:
                    if bias.completes(text):
                        owed = 0.0
                    elif any(m for _, m in bias._open(text)):
                        owed += extra
                    else:
                        total, owed = total - owed, 0.0
                survivors.append((text, total, owed))
                if len(survivors) == k:
                    break
            beams = survivors
            pos += 1
            # Scores only fall as tokens are added, so once no live beam is ahead of the best
            # finished one, none can overtake it.
            if finished and (not beams or max(b[1] for b in beams) < max(f[0] for f in finished)):
                break
        if not finished:
            finished = [(score - owed, text) for text, score, owed in beams]
        _, best = max(finished, key=lambda f: f[0] / (len(f[1]) + 1))
        return self.tokenizer.decode(best), c.languages[tag - c.text]

    def keyword_bias(self, keywords, boost=DEFAULT_BOOST):
        """A KeywordBias for `keywords`: words or phrases, or (word, bonus) pairs to give
        some their own bonus; the rest get `boost`. None when there are no keywords.

        At the start of a transcript a lowercase keyword is also favoured capitalised, as
        Whistle writes a first word ("Kitchen lights off.")."""
        if not keywords:
            return None
        cached = getattr(self, "_bias_cache", None)
        key = (tuple(k if isinstance(k, str) else tuple(k) for k in keywords), boost)
        if cached is not None and cached[0] == key:
            return cached[1]
        encode = self.tokenizer.encode
        starts, start_boosts, paths, path_boosts = [], [], [], []
        for item in keywords:
            word, bonus = (item, boost) if isinstance(item, str) else (item[0], float(item[1]))
            for variant in dict.fromkeys([word, word[:1].upper() + word[1:]]):
                starts.append(encode(variant))
                start_boosts.append(bonus)
            paths.append(encode(" " + word))
            path_boosts.append(bonus)
        bias = KeywordBias(starts, paths, self.tokenizer.word_start, self.cfg.eos, boost,
                           start_boosts=start_boosts, path_boosts=path_boosts)
        self._bias_cache = (key, bias)
        return bias

    def transcribe(self, audio, language=None, keywords=None, boost=DEFAULT_BOOST, beam_size=1):
        """Speech to text. `audio` is a WAV path or 16 kHz mono float samples.

        `language` is one of the checkpoint's languages, or None to detect it from the
        first 30 seconds. `keywords` are words or phrases to favour, written as they
        should appear (or (word, bonus) pairs); `boost` is how strongly. `beam_size` above 1
        uses beam search instead of greedy decoding: usually more accurate, about that many
        times the decoding work. Returns {"text", "language"}.
        """
        if language is not None and language not in self.cfg.languages:
            raise ValueError(f"language must be one of {', '.join(self.cfg.languages)}")
        bias = self.keyword_bias(keywords, boost)
        samples = read_wav(audio) if isinstance(audio, (str, os.PathLike)) else np.asarray(audio, np.float32)
        step = CHUNK_SECONDS * SAMPLE_RATE
        texts = []
        for begin in range(0, max(len(samples), 1), step):
            text, language = self._decode_chunk(samples[begin:begin + step], language, bias, beam_size)
            texts.append(text.strip())
        return {"text": " ".join(t for t in texts if t), "language": language}


def main(args):
    from ..voice import whistle_keywords

    transcriber = Transcriber(args.checkpoint, dtype=args.dtype)
    keywords = whistle_keywords(args)
    for path in args.audio:
        result = transcriber.transcribe(path, language=args.language, keywords=keywords, boost=args.boost,
                                        beam_size=args.beam_size)
        if args.json:
            print(json.dumps({"audio": path, **result}, ensure_ascii=False))
        else:
            prefix = f"{path}: " if len(args.audio) > 1 else ""
            print(f"{prefix}[{result['language']}] {result['text']}")
    sys.stdout.flush()


__all__ = ["LANGUAGES", "KeywordBias", "Transcriber", "WhistleTokenizer", "read_wav", "split_keywords", "main"]
