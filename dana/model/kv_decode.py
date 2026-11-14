"""Step-by-step decoding with a key/value cache.

`SimpleAttentionNetwork.decode` runs every position of the decoder buffer, and projects the whole
encoder output to cross-attention keys and values, for each generated token. Here the cross-attention
keys and values are computed once per input, and each step runs only the newest token, attending to
the cached self-attention keys and values of the tokens before it. The math, weights and dtypes are
those of `Decoder` in architecture.py; models with feed-forward blocks are not handled.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from .architecture import precompute_rope_freqs

_fns = {}


def _norm(x, scale, dtype):
    """ZCRMSNorm."""
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    return ((1 + scale) * x / rms).astype(dtype)


def _heads(x, n, dim):
    """(B, T, n * dim) -> (B, n, T, dim)."""
    return x.reshape(x.shape[0], -1, n, dim).transpose(0, 2, 1, 3)


def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _attend(q, k, v, mask, dtype, out_kernel):
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(q.shape[-1]))
    scores = jax.nn.softmax(jnp.where(mask, scores, jnp.finfo(scores.dtype).min), axis=-1)
    out = jnp.matmul(scores, v)
    out = out.transpose(0, 2, 1, 3).reshape(out.shape[0], -1, out.shape[1] * out.shape[3])
    return out.astype(dtype) @ out_kernel


def _build(config, max_len):
    dt, n_layers = config.jax_dtype, config.num_decoder_layers
    heads, kv_heads = config.num_heads, config.num_kv_heads
    dim, repeats = config.d_model // config.num_heads, config.num_heads // config.num_kv_heads
    cos, sin = precompute_rope_freqs(dim, max_len, config.rope_theta, config.max_seq_len)
    embed_scale = math.sqrt(config.d_model)

    @jax.jit
    def cross_cache(params, encoder_out):
        """Cross-attention keys and values of every decoder layer, (layers, B, heads, S, dim) each."""
        attn = params["decoder"]["layers"]["DecoderBlock_0"]["cross_attn"]

        def layer(k_kernel, v_kernel, k_scale):
            k = _norm(_heads(encoder_out.astype(dt) @ k_kernel, kv_heads, dim), k_scale, dt)
            v = _heads(encoder_out.astype(dt) @ v_kernel, kv_heads, dim)
            return jnp.repeat(k, repeats, axis=1), jnp.repeat(v, repeats, axis=1)

        return jax.vmap(layer)(attn["k_proj"]["kernel"], attn["v_proj"]["kernel"], attn["k_norm"]["scale"])

    @jax.jit
    def step(params, tokens, pos, cache, cross, cross_mask):
        """Logits at decoder position `pos` for `tokens` (B,) there, and the cache with them added."""
        cache_k, cache_v = cache
        cross_k, cross_v = cross
        emb = params["embedding"]["embedding"]
        x = (emb[tokens] * embed_scale).astype(dt)[:, None, :]
        c, s = cos[pos][None, None, None, :], sin[pos][None, None, None, :]
        seen = (jnp.arange(max_len) <= pos)[None, None, None, :]
        blocks = params["decoder"]["layers"]["DecoderBlock_0"]
        for i in range(n_layers):
            p = jax.tree.map(lambda a: a[i], blocks)
            a = p["self_attn"]
            h = _norm(x, p["ZCRMSNorm_0"]["scale"], dt)
            q = _rope(_norm(_heads(h @ a["q_proj"]["kernel"], heads, dim), a["q_norm"]["scale"], dt), c, s)
            k = _rope(_norm(_heads(h @ a["k_proj"]["kernel"], kv_heads, dim), a["k_norm"]["scale"], dt), c, s)
            v = _heads(h @ a["v_proj"]["kernel"], kv_heads, dim)
            cache_k = jax.lax.dynamic_update_slice(cache_k, k[None], (i, 0, 0, pos, 0))
            cache_v = jax.lax.dynamic_update_slice(cache_v, v[None], (i, 0, 0, pos, 0))
            out = _attend(q, jnp.repeat(cache_k[i], repeats, axis=1), jnp.repeat(cache_v[i], repeats, axis=1),
                          seen, dt, a["out_proj"]["kernel"])
            x = x + jax.nn.sigmoid(p["self_attn_gate"]).astype(dt) * out

            a = p["cross_attn"]
            h = _norm(x, p["ZCRMSNorm_1"]["scale"], dt)
            q = _norm(_heads(h @ a["q_proj"]["kernel"], heads, dim), a["q_norm"]["scale"], dt)
            out = _attend(q, cross_k[i], cross_v[i], cross_mask, dt, a["out_proj"]["kernel"])
            x = x + jax.nn.sigmoid(p["cross_attn_gate"]).astype(dt) * out
        x = _norm(x, params["decoder"]["ZCRMSNorm_0"]["scale"], dt)
        return x[:, 0].astype(jnp.float32) @ emb.T, (cache_k, cache_v)

    return cross_cache, step


def stepper(model, params, encoder_out, cross_mask, max_len):
    """A function from the tokens at the next decoder position, (B,), to the logits there, (B, vocab).
    The first call takes the start tokens, at position 0."""
    config = model.config
    if not config.no_feedforward:
        raise ValueError("kv_decode handles decoders without feed-forward blocks only")
    key = (id(model), max_len)
    if key not in _fns:
        _fns[key] = _build(config, max_len)
    cross_cache, step = _fns[key]
    cross = cross_cache(params, encoder_out)
    dim = config.d_model // config.num_heads
    shape = (config.num_decoder_layers, encoder_out.shape[0], config.num_kv_heads, max_len, dim)
    # Keys are kept in float32, as the decoder has them after RoPE; values in the model dtype.
    state = {"pos": 0, "cache": (jnp.zeros(shape, jnp.float32), jnp.zeros(shape, config.jax_dtype))}

    def advance(tokens):
        logits, state["cache"] = step(params, jnp.asarray(np.asarray(tokens, np.int32)), jnp.int32(state["pos"]),
                                      state["cache"], cross, cross_mask)
        state["pos"] += 1
        return logits

    return advance
