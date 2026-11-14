"""LoRA finetuning for Whistle, in float32.

Adapted from cactus-compute/cactus @ bafcf73 (cactus/model/whistle_finetune.py and the
LoRA helpers in cactus/model/finetune.py, Apache-2.0), without quantisation-aware
training, vocabulary growth, new languages or .cact export. Low-rank adapters on the
attention projections are trained on {"audio", "text", "language"} JSONL clips, then
merged into a full checkpoint that `dana transcribe` and `dana listen` load with
--checkpoint.

Usage:
    dana finetune-whistle train.jsonl --validation val.jsonl --out checkpoints/whistle_tuned.safetensors
"""

import json
import math
import os
import time

import numpy as np

from .model import Whistle, WhistleConfig, read_checkpoint, write_checkpoint
from .transcribe import CHUNK_SECONDS, SAMPLE_RATE, Transcriber, WhistleTokenizer, read_wav

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")


def read_clips(path, languages):
    """(samples, text, language) per JSONL line; audio paths are relative to the file."""
    root = os.path.dirname(os.path.abspath(path))
    clips = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text, language = (row.get("text") or "").strip(), row.get("language", "en")
            if language not in languages:
                raise SystemExit(f"{path}:{number}: language must be one of {', '.join(languages)}")
            samples = read_wav(os.path.join(root, row["audio"]))
            if not text or not len(samples) or len(samples) > CHUNK_SECONDS * SAMPLE_RATE:
                raise SystemExit(f"{path}:{number}: needs a transcript and at most {CHUNK_SECONDS} s of audio")
            clips.append((samples, text, language))
    return clips


def lora_target_paths(params, parts):
    """Attention projection kernels inside the scanned layers of `parts` ("encoder", "stack")."""
    from flax.traverse_util import flatten_dict
    return [path for path in flatten_dict(params)
            if path[0] in parts and path[-1] == "kernel" and "layers" in path
            and any(t in path for t in LORA_TARGETS)]


def init_lora(params, paths, rank, key):
    """A ~ N(0, 1/rank^2), B = 0 per target, with the scanned layer axis kept in front."""
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    lora = {}
    for path in paths:
        weight = flat[path]
        key, sub = jax.random.split(key)
        lora[path] = {"A": jax.random.normal(sub, weight.shape[:-1] + (rank,), jnp.float32) / rank,
                      "B": jnp.zeros(weight.shape[:-2] + (rank, weight.shape[-1]), jnp.float32)}
    return lora


def merge_lora(params, lora, scale):
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict, unflatten_dict
    flat = dict(flatten_dict(params))
    for path, adapter in lora.items():
        flat[path] = flat[path] + scale * jnp.matmul(adapter["A"], adapter["B"])
    return unflatten_dict(flat)


def _arrays(items, idx, samples, max_tokens, pad):
    wav = np.zeros((len(idx), samples), np.float32)
    lengths = np.zeros(len(idx), np.int32)
    tokens = np.full((len(idx), max_tokens), pad, np.int32)
    for row, i in enumerate(idx):
        audio, ids = items[i][:2]
        wav[row, :len(audio)], lengths[row], tokens[row, :len(ids)] = audio, len(audio), ids
    return wav, lengths, tokens


def finetune_main(args):
    import jax
    import jax.numpy as jnp
    import optax

    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be at least 1")
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Whistle checkpoint not found: {args.checkpoint}")
    params, saved = read_checkpoint(args.checkpoint)
    cfg = WhistleConfig.from_saved(saved)
    params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), params)
    tokenizer = WhistleTokenizer(cfg)

    def encode(clips):
        return [(audio, [cfg.bos, cfg.tag(lang), *tokenizer.encode(text), cfg.eos], text)
                for audio, text, lang in clips]

    items = encode(read_clips(args.train, cfg.languages))
    validation = encode(read_clips(args.validation, cfg.languages)) if args.validation else []
    everything = items + validation
    if not items:
        raise SystemExit(f"no clips in {args.train}")
    if max(len(ids) for _, ids, _ in everything) > cfg.max_tokens:
        raise SystemExit(f"a transcript is over {cfg.max_tokens} tokens")
    # One padded shape for every batch, so training compiles once; the encoder masks the padding.
    seconds = args.seconds or min(CHUNK_SECONDS, math.ceil(max(len(a) for a, _, _ in everything) / SAMPLE_RATE))
    samples = seconds * SAMPLE_RATE
    if max(len(a) for a, _, _ in everything) > samples:
        raise SystemExit(f"a clip is longer than --seconds {seconds}")
    max_tokens = -(-max(len(ids) for _, ids, _ in everything) // 8) * 8
    minutes = sum(len(a) for a, _, _ in items) / SAMPLE_RATE / 60
    print(f"  {'data':<9} {len(items)} train, {len(validation)} validation clips, {minutes:.1f} min to train on, "
          f"padded to {seconds} s / {max_tokens} tokens")

    model = Whistle(cfg)
    paths = lora_target_paths(params, ("stack",) if args.decoder_only else ("encoder", "stack"))
    scale = args.lora_alpha / args.lora_rank
    lora = init_lora(params, paths, args.lora_rank, jax.random.PRNGKey(args.seed))
    size = sum(int(np.prod(v.shape)) for adapter in lora.values() for v in adapter.values())
    print(f"  {'lora':<9} rank {args.lora_rank}  alpha {args.lora_alpha:g}  {len(paths)} weight groups"
          f"{' (decoder only)' if args.decoder_only else ''}  {size:,} trainable values")

    def loss_fn(lora, wav, lengths, tokens):
        logits = model.apply({"params": merge_lora(params, lora, scale)}, wav, lengths, tokens[:, :-1])
        targets = tokens[:, 1:]
        mask = targets != cfg.pad
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    batch, count = args.batch_size, len(items)
    steps_per_epoch = -(-count // batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(max(1, total_steps // 20), total_steps - 1) if total_steps > 1 else 0
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, max(total_steps, 1))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(lora)

    @jax.jit
    def train_step(lora, opt_state, wav, lengths, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(lora, wav, lengths, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, lora)
        return optax.apply_updates(lora, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)

    def validation_loss():
        # Pad the last batch by repeating clips, so evaluation keeps the training shape.
        losses = []
        for start in range(0, len(validation), batch):
            idx = [(start + i) % len(validation) for i in range(batch)]
            losses.append(float(eval_step(lora, *_arrays(validation, idx, samples, max_tokens, cfg.pad))))
        return float(np.mean(losses))

    print(f"  {'schedule':<9} {total_steps} steps  warmup {warmup}  lr {args.lr:g}  cosine decay  clip 1.0"
          + (f"  val {validation_loss():.4f} before training" if validation else "") + "  (compiling...)",
          flush=True)
    rng = np.random.default_rng(args.seed)
    start_time, step = time.time(), 0
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        order = np.concatenate([order, order[: (-count) % batch]])  # full last batch, same shape
        losses = []
        for start in range(0, count, batch):
            lora, opt_state, loss = train_step(lora, opt_state,
                                               *_arrays(items, order[start:start + batch], samples, max_tokens, cfg.pad))
            losses.append(float(loss))
            step += 1
        note = f"  val {validation_loss():.4f}" if validation else ""
        print(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {np.mean(losses):.4f}{note}"
              f"  ({time.time() - start_time:.0f} s)", flush=True)

    merged = jax.tree.map(np.asarray, merge_lora(params, lora, scale))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_checkpoint(args.out, merged, saved, base=args.checkpoint, train=args.train,
                     lora=json.dumps({"rank": args.lora_rank, "alpha": args.lora_alpha, "epochs": args.epochs,
                                      "lr": args.lr, "decoder_only": bool(args.decoder_only)}))
    print(f"  {'wrote':<9} {args.out}")

    if validation:
        tuned = Transcriber.from_params(merged, cfg)
        right = 0
        for audio, ids, text in validation:
            got = tuned.transcribe(audio, language=cfg.languages[ids[1] - cfg.text])["text"]
            right += got == text
            print(f"  {'ok' if got == text else 'MISS':<9} {got!r}" + ("" if got == text else f"  want {text!r}"))
        print(f"  {'exact':<9} {right}/{len(validation)} validation transcripts")
    print(f"  {'next':<9} dana listen --checkpoint {args.out}")
