"""Building blocks for Whistle, inference-only and in floating point.

Ported from cactus-compute/cactus (Apache-2.0, see LICENSE-APACHE) with the
quantization-aware-training branches removed. Class names and parameter names
are unchanged, so Flax lays out parameters exactly as the official checkpoint
expects.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn.initializers as jinit
import flax.linen as nn

DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}

ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193

HADA_COND_RANK = 8
_HADA_PERM_SEEDS = (11, 13)


def default_init():
    return jinit.normal(stddev=0.02)


def residual_init(num_layers):
    return jinit.normal(stddev=0.02 / math.sqrt(2 * num_layers))


def _conv_identity_init(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype).at[0].set(1.0)


def _res_identity_init(key, shape, dtype=jnp.float32):
    return jnp.broadcast_to(4.0 * jnp.eye(shape[-1], dtype=dtype), shape)


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def precompute_rope_freqs(head_dim, seq_len, theta=100000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    angles = jnp.outer(jnp.arange(seq_len).astype(jnp.float32), freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def _shift_right(x, offset):
    if offset == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[1] = (offset, 0)
    return jnp.pad(x, pad)[:, : x.shape[1]]


def _mask_diag(mask, offset):
    """Per-position flag that the token `offset` steps back is visible under `mask`."""
    m = mask[:, 0]
    T = m.shape[-1]
    if offset >= T:
        return jnp.zeros(m.shape[:-2] + (T,), m.dtype)
    d = jnp.diagonal(m, offset=-offset, axis1=-2, axis2=-1)
    if offset == 0:
        return d
    return jnp.pad(d, ((0, 0), (offset, 0)))


def _rms_unit(x, epsilon=1e-6):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf ** 2, axis=-1, keepdims=True) + epsilon)


def _sinkhorn(logits, iters=20):
    log_K = logits
    for _ in range(iters):
        log_K = log_K - jax.nn.logsumexp(log_K, axis=-1, keepdims=True)
        log_K = log_K - jax.nn.logsumexp(log_K, axis=-2, keepdims=True)
    return jnp.exp(log_K)


def engram_indices(tokens, orders, heads, slots):
    """Hash every n-gram (one hash per order and head) ending at each position into `slots` buckets."""
    u = tokens.astype(jnp.uint32)
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = jnp.full_like(u, jnp.uint32(seed))
            for j in range(order):
                acc = (acc ^ _shift_right(u, j)) * jnp.uint32(_ENGRAM_PRIME)
            acc = acc ^ (acc >> jnp.uint32(15))
            idx.append((acc % jnp.uint32(slots)).astype(jnp.int32))
    return jnp.stack(idx, axis=-1)


class ZCRMSNorm(nn.Module):
    """RMSNorm with a zero-centred scale (weight = 1 + scale)."""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", jinit.zeros, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.epsilon)
        return ((1 + scale) * x / rms).astype(self.dtype)


class Engram(nn.Module):
    """Hashed n-gram memory: looks up table rows by n-gram hash and returns keys and values."""
    d_model: int
    num_tables: int
    slots: int
    sub_dim: int
    num_layers: int
    conv_dilation: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, indices, ngram_ok, tap_ok):
        tables = self.param("embedding", default_init(), (self.num_tables, self.slots, self.sub_dim))
        fetched = tables[jnp.arange(self.num_tables), indices] * ngram_ok[..., None]
        e = fetched.reshape(*indices.shape[:2], self.num_tables * self.sub_dim).astype(self.dtype)
        k = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                     kernel_init=default_init(), name="key_proj")(e)
        v = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                     kernel_init=residual_init(self.num_layers), name="value_proj")(e)
        taps = self.param("taps", _conv_identity_init, (ENGRAM_CONV_TAPS, self.d_model)).astype(self.dtype)
        v = sum(taps[j] * _shift_right(v, j * self.conv_dilation) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        return k, v


class MultiHeadAttention(nn.Module):
    """Gated grouped-query attention with QK norm, RoPE and optional causal QKV taps."""
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True
    qk_head_dim: int = 0
    v_head_dim: int = 0
    qkv_conv_taps: int = 0

    @nn.compact
    def __call__(self, x, mask=None, rope=None):
        qk_hd = self.qk_head_dim or self.d_model // self.num_heads
        v_hd = self.v_head_dim or self.d_model // self.num_heads
        out_dim = self.num_heads * v_hd
        B = x.shape[0]

        dense = lambda n, name: nn.Dense(n, dtype=self.dtype, use_bias=False,  # noqa: E731
                                         kernel_init=default_init(), name=name)
        q = dense(self.num_heads * qk_hd, "q_proj")(x)
        k = dense(self.num_kv_heads * qk_hd, "k_proj")(x)
        v = dense(self.num_kv_heads * v_hd, "v_proj")(x)

        if self.qkv_conv_taps:
            n_taps = self.qkv_conv_taps
            qt = self.param("q_taps", _conv_identity_init, (n_taps, self.num_heads * qk_hd)).astype(self.dtype)
            kt = self.param("k_taps", _conv_identity_init, (n_taps, self.num_kv_heads * qk_hd)).astype(self.dtype)
            vt = self.param("v_taps", _conv_identity_init, (n_taps, self.num_kv_heads * v_hd)).astype(self.dtype)
            tap_ok = [None] + [_mask_diag(mask, j)[..., None].astype(q.dtype) if mask is not None else None
                               for j in range(1, n_taps)]

            def tapped(z, j):
                shifted = _shift_right(z, j)
                return shifted if tap_ok[j] is None else shifted * tap_ok[j]

            q = sum(qt[j] * tapped(q, j) for j in range(n_taps))
            k = sum(kt[j] * tapped(k, j) for j in range(n_taps))
            v = sum(vt[j] * tapped(v, j) for j in range(n_taps))

        q = q.reshape(B, -1, self.num_heads, qk_hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_kv_heads, qk_hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_kv_heads, v_hd).transpose(0, 2, 1, 3)

        q = ZCRMSNorm(dtype=self.dtype, name="q_norm")(q)
        k = ZCRMSNorm(dtype=self.dtype, name="k_norm")(k)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.flash:
            fused_hd = max(qk_hd, v_hd)
            qf = q * jnp.asarray(math.sqrt(fused_hd / qk_hd), q.dtype)
            pad_qk = fused_hd - qk_hd
            if pad_qk:
                qf = jnp.pad(qf, ((0, 0), (0, 0), (0, 0), (0, pad_qk)))
                k = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad_qk)))
            pad_v = fused_hd - v_hd
            vf = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_v))) if pad_v else v
            impl = "cudnn" if jax.default_backend() == "gpu" and q.dtype in (jnp.bfloat16, jnp.float16) else None
            out = jax.nn.dot_product_attention(qf.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
                                               vf.transpose(0, 2, 1, 3), mask=mask, implementation=impl)
            out = out[..., :v_hd].reshape(B, -1, out_dim)
        else:
            repeats = self.num_heads // self.num_kv_heads
            if repeats > 1:
                k = jnp.repeat(k, repeats, axis=1)
                v = jnp.repeat(v, repeats, axis=1)
            weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(qk_hd))
            if mask is not None:
                weights = jnp.where(mask, weights, jnp.finfo(weights.dtype).min)
            out = jnp.matmul(nn.softmax(weights, axis=-1), v).transpose(0, 2, 1, 3).reshape(B, -1, out_dim)

        out = out * nn.sigmoid(dense(out_dim, "gate_proj")(x))
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                        kernel_init=residual_init(self.num_layers), name="out_proj")(out)


def _walsh_matrix(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return jnp.asarray(H / np.sqrt(n))


def _walsh_init(size):
    w = _walsh_matrix(size)
    return lambda key, shape: jnp.asarray(w)


def _hada_blocks(n):
    b = 1 << ((n - 1).bit_length() // 2)
    return b, n // b


def _hada_perms(n):
    return tuple(jnp.asarray(np.random.RandomState(s).permutation(n)) for s in _HADA_PERM_SEEDS)


def _kron_apply(z, a, b):
    lead = z.shape[:-1]
    z = z.reshape(*lead, a.shape[0], b.shape[0])
    z = jnp.einsum("...ij,ik,jl->...kl", z, a, b)
    return z.reshape(*lead, a.shape[0] * b.shape[0])


class HadamardMLP(nn.Module):
    """Feed-forward layer built from Kronecker-factored Walsh-Hadamard mixes and fixed permutations."""
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        n = 1 << (self.d_model - 1).bit_length()
        ba, bb = _hada_blocks(n)
        p1, p2 = _hada_perms(n)
        factors = [
            (self.param(f"w{s}a", _walsh_init(ba), (ba, ba)).astype(self.dtype),
             self.param(f"w{s}b", _walsh_init(bb), (bb, bb)).astype(self.dtype))
            for s in (1, 2, 3)
        ]
        d1 = self.param("d1", jinit.ones, (n,)).astype(self.dtype)
        d2 = self.param("d2", jinit.ones, (n,)).astype(self.dtype)
        b2 = self.param("b2", jinit.zeros, (n,)).astype(self.dtype)
        d3 = self.param("d3", jinit.ones, (n,)).astype(self.dtype)
        d4 = self.param("d4", jinit.constant(0.02), (n,)).astype(self.dtype)
        cond_v = self.param("cond_v", default_init(), (self.d_model, HADA_COND_RANK)).astype(self.dtype)
        cond_u = self.param("cond_u", jinit.zeros, (HADA_COND_RANK, n)).astype(self.dtype)

        cond = 1 + nn.softmax(x @ cond_v, axis=-1) @ cond_u
        pad = n - self.d_model
        z = jnp.pad(x, ((0, 0), (0, 0), (0, pad))) if pad else x
        z = _kron_apply(d1 * z, *factors[0])[..., p1]
        z = _kron_apply(nn.silu(d2 * cond * z + b2), *factors[1])[..., p2]
        z = _kron_apply(d3 * z, *factors[2])
        return (d4 * z)[..., : self.d_model]
