"""Label hand-written queries with the official model on short inputs (each tool's own pool)."""
import glob, json, os, sys, time
from dana import SimpleAttentionNetwork, load_checkpoint, get_tokenizer
from dana.model.run import generate_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pools = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/examples/tools/*.json"))}
owner = {t["name"]: p for p, tools in pools.items() for t in tools}

rows = []
for line in open(f"{ROOT}/data/longctx/queries.txt"):
    name, *qs = line.rstrip("\n").split("|")
    for i, q in enumerate(qs):
        rows.append({"tool": name, "pool": owner[name], "query": q, "split": "test" if i == 2 else "train"})

params, config = load_checkpoint(f"{ROOT}/checkpoints/dana.pkl")
model, tok = SimpleAttentionNetwork(config), get_tokenizer()
out = open(f"{ROOT}/data/longctx/labeled.jsonl", "w")
kept, t0 = 0, time.time()
for i in range(0, len(rows), 32):
    chunk = rows[i:i + 32]
    preds = generate_batch(model, params, tok, [r["query"] for r in chunk],
                           [json.dumps(pools[r["pool"]]) for r in chunk], max_enc_len=2048, max_gen_len=160)
    for r, p in zip(chunk, preds):
        try:
            calls = json.loads(p)
        except ValueError:
            calls = []
        r["answers"] = json.dumps(calls)
        r["ok"] = [c.get("name") for c in calls] == [r["tool"]]
        kept += r["ok"]
        out.write(json.dumps(r) + "\n")
    out.flush()
    print(f"{i + len(chunk)}/{len(rows)} kept {kept} ({time.time() - t0:.0f}s)", flush=True)
