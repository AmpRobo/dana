"""Check dana.whistle against the original cactus Whistle on shared random weights.

Builds both models with the same config, initialises the original, perturbs every
parameter (so zero-initialised gates are exercised), then compares the parameter
layout, encoder output and decoder logits. Needs a cactus checkout, not its weights.

  python scripts/whistle_parity.py --cactus ../cactus
  python scripts/whistle_parity.py --cactus ../cactus --small
"""
import argparse
import dataclasses
import os
import sys
import types

import numpy as np
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _import_cactus_whistle(cactus_root):
    # Import cactus.model.whistle without running cactus/__init__.py (runtime and telemetry).
    pkg = types.ModuleType("cactus")
    pkg.__path__ = [os.path.join(cactus_root, "cactus")]
    sys.modules["cactus"] = pkg
    from cactus.model import whistle
    return whistle


def _shapes(tree):
    return {"/".join(str(k.key) for k in path): leaf.shape
            for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cactus", default=os.path.join(os.path.dirname(ROOT), "cactus"))
    ap.add_argument("--small", action="store_true", help="Shrink the model for a quick run")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    orig = _import_cactus_whistle(os.path.abspath(args.cactus))
    from dana.whistle import model as port

    overrides = dict(dim=128, enc_layers=2, dec_layers=4, sites=(1, 3), slots=512, stem=16) if args.small else {}
    ocfg = orig.WhistleConfig(remat=False, **overrides)
    pcfg = port.WhistleConfig(**overrides)
    assert dataclasses.asdict(ocfg).keys() == dataclasses.asdict(pcfg).keys()

    B, n = 2, int(args.seconds * ocfg.rate)
    rng = np.random.default_rng(0)
    wav = rng.standard_normal((B, n)).astype(np.float32) * 0.1
    lengths = np.array([n, n * 2 // 3], np.int32)
    tokens = rng.integers(3, ocfg.vocab, (B, 24)).astype(np.int32)
    tokens[:, 0] = ocfg.bos

    omodel, pmodel = orig.Whistle(ocfg), port.Whistle(pcfg)
    params = omodel.init(jax.random.PRNGKey(0), wav, lengths, tokens)["params"]
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    params = jax.tree.unflatten(treedef, [x + 0.05 * jax.random.normal(k, x.shape, x.dtype)
                                          for x, k in zip(leaves, keys)])

    pinit = pmodel.init(jax.random.PRNGKey(0), wav, lengths, tokens)["params"]
    oshapes, pshapes = _shapes(params), _shapes(pinit)
    missing, extra = sorted(oshapes.keys() - pshapes.keys()), sorted(pshapes.keys() - oshapes.keys())
    mismatched = sorted(k for k in oshapes.keys() & pshapes.keys() if oshapes[k] != pshapes[k])
    print(f"params: {len(oshapes)} tensors, {sum(np.prod(s) for s in oshapes.values()):,} values; "
          f"missing={len(missing)} extra={len(extra)} shape_mismatch={len(mismatched)}")
    for k in (missing + extra + mismatched)[:10]:
        print("   ", k, oshapes.get(k), pshapes.get(k))

    oenc, oframes = omodel.apply({"params": params}, wav, lengths, method=orig.Whistle.encode)
    penc, pframes = pmodel.apply({"params": params}, wav, lengths, method=port.Whistle.encode)
    ologits = omodel.apply({"params": params}, wav, lengths, tokens)
    plogits = pmodel.apply({"params": params}, wav, lengths, tokens)

    enc_diff = float(jnp.max(jnp.abs(oenc - penc)))
    logit_diff = float(jnp.max(jnp.abs(ologits - plogits)))
    scale = float(jnp.max(jnp.abs(ologits)))
    same_argmax = bool(jnp.all(ologits.argmax(-1) == plogits.argmax(-1)))
    print(f"encoder: frames {np.asarray(oframes).tolist()} vs {np.asarray(pframes).tolist()}, max |diff| {enc_diff:.3g}")
    print(f"logits:  shape {tuple(plogits.shape)}, max |diff| {logit_diff:.3g} (max |logit| {scale:.3g}), "
          f"same argmax: {same_argmax}")
    ok = (not missing and not extra and not mismatched and np.array_equal(oframes, pframes)
          and logit_diff <= 1e-4 * max(scale, 1.0) and same_argmax)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
