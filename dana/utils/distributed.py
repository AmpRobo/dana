import os
import pickle
import threading
import jax
import jax.numpy as jnp
import numpy as np

def _require_checkpoint(path):
    """Return `path` if the checkpoint exists on disk, else raise FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _flatten_params(tree, prefix=()):
    """Flatten a nested dict of params into {tuple-path: leaf}."""
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten_params(v, prefix + (k,)))
    else:
        out[prefix] = tree
    return out


def _unflatten_params(flat):
    """Inverse of _flatten_params: rebuild a nested dict."""
    root = {}
    for path, leaf in flat.items():
        node = root
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = leaf
    return root


def partial_load_params(init_params, loaded_params):
    """Merge `loaded_params` into `init_params` by matching path + shape.

    For every leaf in `init_params`:
      - if the same path exists in `loaded_params` and shapes agree,
        replace it with the loaded value (cast to the init leaf's dtype);
      - otherwise keep the init leaf (random init).

    Extra leaves in `loaded_params` (not present in `init_params`) are ignored.

    Returns (merged_params, stats_dict).
    """
    init_flat = _flatten_params(init_params)
    loaded_flat = _flatten_params(loaded_params)

    merged = {}
    n_loaded = 0
    n_random = 0
    shape_mismatches = []
    missing_in_ckpt = []
    for path, init_leaf in init_flat.items():
        loaded_leaf = loaded_flat.get(path)
        if loaded_leaf is None:
            merged[path] = init_leaf
            n_random += 1
            missing_in_ckpt.append(path)
            continue
        if tuple(loaded_leaf.shape) != tuple(init_leaf.shape):
            merged[path] = init_leaf
            n_random += 1
            shape_mismatches.append((path, tuple(loaded_leaf.shape), tuple(init_leaf.shape)))
            continue
        merged[path] = jnp.asarray(loaded_leaf, dtype=init_leaf.dtype)
        n_loaded += 1

    extra_in_ckpt = [p for p in loaded_flat.keys() if p not in init_flat]

    stats = {
        "loaded": n_loaded,
        "random_init": n_random,
        "shape_mismatches": shape_mismatches,
        "missing_in_ckpt": missing_in_ckpt,
        "extra_in_ckpt": extra_in_ckpt,
    }
    return _unflatten_params(merged), stats


def load_pretrained_params(path_or_name):
    """pickle.load a local checkpoint, return its params dict."""
    local = _require_checkpoint(path_or_name)
    with open(local, "rb") as f:
        data = pickle.load(f)
    return data["params"], data.get("config", None), local


def _replicate(tree):
    """Replicate a pytree across all local devices (multi-host safe)."""
    devices = np.array(jax.local_devices())
    n = len(devices)
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))

    def _rep(x):
        x = jnp.asarray(x)
        return jax.device_put(jnp.broadcast_to(x, (n,) + x.shape), sharding)

    return jax.tree.map(_rep, tree)


def _unreplicate(tree):
    """Get a single copy from a replicated pytree (multi-host safe).

    Uses addressable_shards instead of x[0] indexing, which fails
    in multi-host pmap because the array spans devices on other hosts.
    Each shard has shape (1, ...) — we index [0] to strip the leading dim.
    """
    def _get_first(x):
        if hasattr(x, 'addressable_shards') and x.addressable_shards:
            shard = x.addressable_shards[0].data
            # Shard has shape (1, ...) from the pmap partition; strip it
            return jax.device_get(shard[0]) if shard.ndim > 0 else jax.device_get(shard)
        return x
    return jax.tree.map(_get_first, tree)


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])

