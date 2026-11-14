"""Fine-tune Dana 1 for a 4096-token encoder on data/longctx, and evaluate by input length.

  python scripts/longctx_finetune.py eval  --checkpoint checkpoints/dana.pkl
  python scripts/longctx_finetune.py train --epochs 2 --batch-size 8
"""
import argparse, json, os, pickle, sys, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data/longctx"
MAX_ENC, MAX_DEC, GEN = 4096, 256, 160


def load(name):
    return [json.loads(l) for l in open(f"{DATA}/{name}.jsonl")]


def evaluate(checkpoint, max_enc_len=MAX_ENC, buckets=None, limit=None, batch=16):
    from dana import SimpleAttentionNetwork, load_checkpoint, get_tokenizer
    from dana.model.run import generate_batch

    params, config = load_checkpoint(checkpoint)
    model, tok = SimpleAttentionNetwork(config), get_tokenizer()
    by_bucket = defaultdict(list)
    for e in load("test"):
        if not buckets or e["bucket"] in buckets:
            by_bucket[e["bucket"]].append(e)
    results = {}
    for b, rows in by_bucket.items():
        rows = rows[:limit] if limit else rows
        name_ok = exact = 0
        t0 = time.time()
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            preds = generate_batch(model, params, tok, [r["query"] for r in chunk], [r["tools"] for r in chunk],
                                   max_gen_len=GEN, max_enc_len=max_enc_len)
            for r, p in zip(chunk, preds):
                try:
                    calls = json.loads(p)
                except ValueError:
                    calls = []
                calls = calls if isinstance(calls, list) else [calls]
                name_ok += [c.get("name") for c in calls if isinstance(c, dict)] == [r["tool"]]
                exact += json.dumps(calls, sort_keys=True) == json.dumps(json.loads(r["answers"]), sort_keys=True)
        n = len(rows)
        results[b] = {"n": n, "tool_acc": round(name_ok / n, 4), "exact": round(exact / n, 4)}
        print(f"  {b:8s} n={n} tool_acc={name_ok / n:.1%} exact={exact / n:.1%} ({time.time() - t0:.0f}s)", flush=True)
    return results


def finetune(args):
    import shutil, tempfile
    from datasets import Dataset
    from dana.dataset import dataset as data_mod
    from dana.dataset.dataset import get_tokenizer
    from dana.training.finetune import _write_split
    from dana.training.train import train

    strip = lambda rows: [{k: e[k] for k in ("query", "tools", "answers")} for e in rows]
    train_rows, val_rows = strip(load("train")), strip(load("val"))
    with open(args.base, "rb") as f:
        cfg = pickle.load(f)["config"]

    cache_dir = tempfile.mkdtemp(prefix="dana-longctx-")
    original = data_mod.CACHE_DIR
    data_mod.CACHE_DIR = cache_dir
    try:
        tok = get_tokenizer()
        n_train = _write_split("train", train_rows, tok, cache_dir, MAX_ENC, MAX_DEC)
        n_val = _write_split("val", val_rows, tok, cache_dir, MAX_ENC, MAX_DEC)
        print(f"Tokenized {n_train} train / {n_val} val")
        steps = max(1, n_train // args.batch_size)
        train(argparse.Namespace(
            name=args.name, checkpoint=args.base, init_from=None,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, muon_lr=args.muon_lr,
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            num_kv_heads=cfg.get("num_kv_heads", cfg["num_heads"]),
            num_layers=cfg["num_encoder_layers"], num_dec_layers=cfg["num_decoder_layers"],
            d_ff=cfg.get("d_ff", cfg["d_model"] * 4),
            max_enc_len=MAX_ENC, max_dec_len=MAX_DEC, max_samples=None,
            warmup_ratio=0.05, decay_ratio=0.3, wandb=False, dtype=cfg.get("dtype", "bfloat16"),
            checkpoint_dir=f"{ROOT}/checkpoints", seed=42,
            eval_every=steps * args.epochs + 1, max_eval_samples=n_val,
            contrastive_weight=0.1, contrastive_dim=cfg.get("contrastive_dim", 128),
            num_memory_slots=cfg.get("num_memory_slots", 64),
            w_name=2.0, w_value=4.0, w_key=1.5, val_ds=Dataset.from_list(val_rows),
        ))
    finally:
        data_mod.CACHE_DIR = original
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", default=f"{ROOT}/checkpoints/dana.pkl")
    e.add_argument("--max-enc-len", type=int, default=MAX_ENC)
    e.add_argument("--buckets", nargs="*")
    e.add_argument("--limit", type=int)
    e.add_argument("--out")
    t = sub.add_parser("train")
    t.add_argument("--base", default=f"{ROOT}/checkpoints/dana.pkl")
    t.add_argument("--name", default="dana_4k")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--muon-lr", type=float, default=0.005)
    a = p.parse_args()
    if a.cmd == "eval":
        r = evaluate(a.checkpoint, a.max_enc_len, a.buckets, a.limit)
        if a.out:
            json.dump(r, open(a.out, "w"), indent=1)
    else:
        finetune(a)
