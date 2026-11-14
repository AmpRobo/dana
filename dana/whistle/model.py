"""Whistle speech-to-text model (encoder-decoder over log-mel features).

Ported from cactus-compute/cactus (Apache-2.0, see LICENSE-APACHE), inference
only and in floating point. Parameter names match the official checkpoint.
"""

import math
from dataclasses import dataclass, fields
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn.initializers as jinit
import flax.linen as nn

from .layers import (
    DTYPE_MAP, ENGRAM_CONV_TAPS, Engram, HadamardMLP, MultiHeadAttention, ZCRMSNorm,
    _mask_diag, _res_identity_init, _rms_unit, _sinkhorn, default_init, engram_indices,
    make_causal_mask, precompute_rope_freqs, residual_init,
)

LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")


@dataclass(frozen=True)
class WhistleConfig:
    text: int = 8192
    languages: tuple = LANGUAGES
    dim: int = 512
    kv_heads: int = 2
    qk_dim: int = 48
    v_dim: int = 64
    enc_layers: int = 8
    dec_layers: int = 8
    conv_kernel: int = 9
    taps: int = 3
    lanes: int = 4
    enc_lanes: int = 4
    sites: tuple = (3, 7)
    orders: tuple = (2, 3)
    slots: int = 18432
    hashes: int = 2
    stem: int = 128
    stem_stages: int = 3
    rope: float = 100000.0
    mels: int = 80
    rate: int = 16000
    fft: int = 512
    win: int = 400
    hop: int = 160
    max_tokens: int = 320
    pad: int = 0
    eos: int = 1
    bos: int = 2
    dtype: str = "float32"
    flash: bool = True
    remat: bool = False
    scan_unroll: int = 1

    @classmethod
    def from_saved(cls, saved, **overrides):
        known = {f.name for f in fields(cls)} - {"dtype", "flash", "remat", "scan_unroll"}
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in known}
        return cls(**kwargs, **overrides)

    @property
    def vocab(self):
        """Text pieces followed by one tag token per language."""
        return self.text + len(self.languages)

    @property
    def heads(self):
        return self.dim // self.v_dim

    @property
    def jax_dtype(self):
        return DTYPE_MAP[self.dtype]

    @property
    def engram_sub_dim(self):
        return self.dim // (len(self.orders) * self.hashes)

    def tag(self, language):
        return self.text + self.languages.index(language)


def mel_matrix(cfg):
    def to_mel(f):
        return np.where(f < 1000.0, f * 3.0 / 200.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) * 27.0 / np.log(6.4))

    def to_hz(m):
        return np.where(m < 15.0, m * 200.0 / 3.0, 1000.0 * np.exp((m - 15.0) * np.log(6.4) / 27.0))

    points = to_hz(np.linspace(0.0, to_mel(np.array(cfg.rate / 2.0)), cfg.mels + 2))
    freqs = np.linspace(0.0, cfg.rate / 2, cfg.fft // 2 + 1)[:, None]
    lo, mid, hi = points[:-2], points[1:-1], points[2:]
    bank = np.maximum(0.0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    return (bank * 2.0 / (hi - lo)).astype(np.float32)


def features(cfg, wav, lengths):
    """Per-utterance normalised log-mel features (B, frames, mels) and frame counts."""
    wav = wav.astype(jnp.float32) * (jnp.arange(wav.shape[1]) < lengths[:, None])
    window = np.hanning(cfg.win).astype(np.float32)
    _, _, z = jax.scipy.signal.stft(
        wav, window=window, nperseg=cfg.win, noverlap=cfg.win - cfg.hop, nfft=cfg.fft, boundary="zeros", padded=False)
    power = jnp.swapaxes(jnp.abs(z * window.sum()) ** 2, 1, 2)
    x = jnp.log(power @ mel_matrix(cfg) + 2.0 ** -24)
    lengths = lengths // cfg.hop
    mask = (jnp.arange(x.shape[1]) < lengths[:, None])[..., None]
    n = jnp.maximum(lengths, 1)[:, None, None]
    mean = (x * mask).sum(1, keepdims=True) / n
    std = jnp.sqrt((((x - mean) * mask) ** 2).sum(1, keepdims=True) / jnp.maximum(n - 1, 1))
    return (x - mean) / (std + 1e-5) * mask, lengths


def sinusoids(cfg, frames):
    ang = jnp.arange(frames)[:, None] * jnp.exp(-np.log(10000.0) / (cfg.dim // 2 - 1) * jnp.arange(cfg.dim // 2))
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1)


def _cross_attend(q, k, v, mem_mask):
    s = jnp.einsum("bshd,bthd->bhst", q.astype(jnp.float32), k.astype(jnp.float32)) / math.sqrt(q.shape[-1])
    s = jnp.where(mem_mask[:, None, None, :], s, jnp.finfo(jnp.float32).min)
    return jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(s, -1), v.astype(jnp.float32)).astype(v.dtype)


class Stem(nn.Module):
    """Depthwise-separable 2D conv stack that downsamples the mel frames by 2 per stage."""
    cfg: WhistleConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, lengths):
        c = self.cfg

        def down(x, w, lengths):
            x = jax.lax.conv_general_dilated(
                x, w.astype(x.dtype), (2, 2), ((1, 1), (1, 1)),
                dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=x.shape[-1])
            lengths = (lengths + 1) // 2
            return x * (jnp.arange(x.shape[1]) < lengths[:, None])[..., None, None], lengths

        x, lengths = down(x[..., None], self.param("w", default_init(), (3, 3, 1, c.stem)), lengths)
        for i in range(1, c.stem_stages):
            x, lengths = down(jax.nn.silu(x), self.param(f"dw_{i}", default_init(), (3, 3, 1, c.stem)), lengths)
            x = nn.Dense(c.stem, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name=f"pw_{i}")(x)
        x = jnp.swapaxes(jax.nn.silu(x), 2, 3)
        x = x.reshape(*x.shape[:2], -1)
        return nn.Dense(c.dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="out")(x), lengths


class EncoderBlock(nn.Module):
    """Conformer-style block: half MLP, self-attention, depthwise conv, half MLP."""
    cfg: WhistleConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask, valid, rope):
        c, dt = self.cfg, self.dtype
        x = x + 0.5 * HadamardMLP(c.dim, dt, name="hadamard_mlp_0")(ZCRMSNorm(dtype=dt, name="pre_hada_norm_0")(x))

        skip = x
        h = ZCRMSNorm(dtype=dt, name="ZCRMSNorm_0")(x)
        h = MultiHeadAttention(c.heads, c.kv_heads, c.dim, c.enc_layers, dt, c.flash, c.qk_dim, c.v_dim, 0,
                               name="self_attn")(h, mask=mask, rope=rope)
        h = ZCRMSNorm(dtype=dt, name="post_attn_norm")(h)
        x = skip + nn.sigmoid(self.param("attn_gate", jinit.zeros, ())).astype(dt) * h

        h = ZCRMSNorm(dtype=dt, name="conv_norm")(x)
        h = nn.Dense(2 * c.dim, dtype=dt, use_bias=False, kernel_init=default_init(), name="pw1")(h)
        h = jax.nn.glu(h) * valid
        dw = self.param("dw", default_init(), (c.conv_kernel, 1, c.dim)).astype(h.dtype)
        h = jax.lax.conv_general_dilated(h, dw, (1,), "SAME", dimension_numbers=("NWC", "WIO", "NWC"),
                                         feature_group_count=c.dim)
        h = jax.nn.silu(ZCRMSNorm(dtype=dt, name="conv_out_norm")(h))
        x = x + nn.Dense(c.dim, dtype=dt, use_bias=False, kernel_init=residual_init(c.enc_layers), name="pw2")(h)

        return x + 0.5 * HadamardMLP(c.dim, dt, name="hadamard_mlp")(ZCRMSNorm(dtype=dt, name="pre_hada_norm")(x))


class CrossAttention(nn.Module):
    cfg: WhistleConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mem, mem_mask):
        c = self.cfg
        heads, qk_hd, v_hd = c.heads, c.qk_dim, c.v_dim
        B = x.shape[0]

        dense = partial(nn.Dense, dtype=self.dtype, use_bias=False, kernel_init=default_init())
        q = dense(heads * qk_hd, name="q_proj")(x).reshape(B, -1, heads, qk_hd)
        k = dense(heads * qk_hd, name="k_proj")(mem).reshape(B, -1, heads, qk_hd)
        v = dense(heads * v_hd, name="v_proj")(mem).reshape(B, -1, heads, v_hd)
        q = ZCRMSNorm(dtype=self.dtype, name="q_norm")(q.transpose(0, 2, 1, 3))
        k = ZCRMSNorm(dtype=self.dtype, name="k_norm")(k.transpose(0, 2, 1, 3))
        v = v.transpose(0, 2, 1, 3)

        if c.flash:
            fused = max(qk_hd, v_hd)
            qf = q * jnp.asarray(math.sqrt(fused / qk_hd), q.dtype)
            if fused > qk_hd:
                qf = jnp.pad(qf, ((0, 0), (0, 0), (0, 0), (0, fused - qk_hd)))
                k = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, fused - qk_hd)))
            vf = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, fused - v_hd))) if fused > v_hd else v
            mask = jnp.broadcast_to(mem_mask[:, None, None, :], (B, 1, qf.shape[2], k.shape[2]))
            out = jax.nn.dot_product_attention(qf.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
                                               vf.transpose(0, 2, 1, 3), mask=mask)
            out = out[..., :v_hd].reshape(B, -1, heads * v_hd)
        else:
            out = _cross_attend(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), mem_mask)
            out = out.reshape(B, -1, heads * v_hd)

        out = out * nn.sigmoid(dense(heads * v_hd, name="gate_proj")(x))
        return nn.Dense(c.dim, dtype=self.dtype, use_bias=False, kernel_init=residual_init(c.dec_layers),
                        name="out_proj")(out)


class DecoderBlock(nn.Module):
    """Engram injection, causal self-attention, cross-attention to audio, Hadamard MLP."""
    cfg: WhistleConfig
    dtype: jnp.dtype = jnp.float32

    def _gate(self, name):
        return nn.sigmoid(self.param(name, jinit.zeros, ())).astype(self.dtype)

    @nn.compact
    def __call__(self, x, mask=None, rope=None, engram_kv=None, site_flags=None, mem=None, mem_mask=None):
        c, dt = self.cfg, self.dtype
        if engram_kv is not None:
            ek, ev = engram_kv
            alpha = nn.sigmoid(jnp.einsum("btd,sbtd->sbt", _rms_unit(x), _rms_unit(ek)) / math.sqrt(c.dim))
            x = x + jnp.einsum("s,sbt,sbtd->btd", site_flags.astype(jnp.float32), alpha,
                               ev.astype(jnp.float32)).astype(x.dtype)

        skip = x
        x = ZCRMSNorm(dtype=dt)(x)
        x = MultiHeadAttention(c.heads, c.kv_heads, c.dim, c.dec_layers, dt, c.flash, qk_head_dim=c.qk_dim,
                               v_head_dim=c.v_dim, qkv_conv_taps=c.taps,
                               name="self_attn")(x, mask=mask, rope=rope)
        x = ZCRMSNorm(dtype=dt, name="post_attn_norm")(x)
        x = skip + self._gate("attn_gate") * x

        skip = x
        x = ZCRMSNorm(dtype=dt, name="cross_norm")(x)
        x = CrossAttention(c, dt, name="cross_attn")(x, mem, mem_mask)
        x = ZCRMSNorm(dtype=dt, name="post_cross_norm")(x)
        x = skip + self._gate("cross_gate") * x

        skip = x
        x = ZCRMSNorm(dtype=dt, name="pre_hada_norm")(x)
        return skip + HadamardMLP(c.dim, dt, name="hadamard_mlp")(x)


class _LaneBody(nn.Module):
    make: object
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, xs, block_kwargs):
        site_flags, hc = xs
        block = self.make(name="block")
        extra = {} if site_flags is None else {"site_flags": site_flags}
        B, T, n, C = x.shape
        xf = x.astype(jnp.float32)
        nx = _rms_unit(x.reshape(B, T, n * C))
        hpre = nn.sigmoid(hc["a_pre"] * (nx @ hc["phi_pre"].astype(jnp.float32)) + hc["b_pre"] + hc["pre_off"])
        u = jnp.einsum("btn,btnc->btc", hpre, xf).astype(self.dtype)
        y = block(u, **block_kwargs, **extra) - u
        hpost = 2 * nn.sigmoid(hc["a_post"] * (nx @ hc["phi_post"].astype(jnp.float32)) + hc["b_post"] + hc["post_off"])
        hres = _sinkhorn(hc["a_res"] * (nx @ hc["phi_res"].astype(jnp.float32)).reshape(B, T, n, n) + hc["b_res"])
        out = jnp.einsum("btij,btjc->btic", hres, xf) + hpost[..., None] * y.astype(jnp.float32)[:, :, None, :]
        return out.astype(self.dtype), None


class LaneStack(nn.Module):
    """Scanned layer stack over `lanes` parallel residual streams mixed by Sinkhorn matrices."""
    num_layers: int
    lanes: int
    d_model: int
    make: object
    engram_layers: tuple = ()
    dtype: jnp.dtype = jnp.float32
    remat: bool = False
    unroll: int = 1

    @nn.compact
    def __call__(self, x, **block_kwargs):
        x = x.astype(self.dtype)
        site_flags = None
        if block_kwargs.get("engram_kv") is not None:
            flags = np.zeros((self.num_layers, len(self.engram_layers)), np.float32)
            for s, layer in enumerate(self.engram_layers):
                flags[layer, s] = 1.0
            site_flags = jnp.asarray(flags)

        n, L, nC = self.lanes, self.num_layers, self.lanes * self.d_model
        lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
        hc = {
            "phi_pre": self.param("mhc_phi_pre", default_init(), (L, nC, n)),
            "phi_post": self.param("mhc_phi_post", default_init(), (L, nC, n)),
            "phi_res": self.param("mhc_phi_res", default_init(), (L, nC, n * n)),
            "b_pre": self.param("mhc_b_pre", jinit.zeros, (L, n)),
            "b_post": self.param("mhc_b_post", jinit.zeros, (L, n)),
            "b_res": self.param("mhc_b_res", _res_identity_init, (L, n, n)),
            "a_pre": self.param("mhc_a_pre", jinit.constant(0.01), (L,)),
            "a_post": self.param("mhc_a_post", jinit.constant(0.01), (L,)),
            "a_res": self.param("mhc_a_res", jinit.constant(0.01), (L,)),
            "pre_off": jnp.asarray(8 * lane - 4),
            "post_off": jnp.asarray(-4 * (1 - lane)),
        }
        x = jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], n, x.shape[-1]))
        scan = nn.scan(nn.remat(_LaneBody) if self.remat else _LaneBody, variable_axes={"params": 0},
                       split_rngs={"params": True}, length=L, in_axes=(0, nn.broadcast), unroll=self.unroll)
        x, _ = scan(self.make, self.dtype, name="layers")(x, (site_flags, hc), block_kwargs)
        return ZCRMSNorm(dtype=self.dtype, name="final_norm")(jnp.mean(x, axis=2))


class Whistle(nn.Module):
    cfg: WhistleConfig

    def setup(self):
        c, dt = self.cfg, self.cfg.jax_dtype
        self.stem = Stem(c, dt)
        self.encoder = LaneStack(c.enc_layers, c.enc_lanes, c.dim, partial(EncoderBlock, c, dt), dtype=dt,
                                 remat=c.remat, unroll=c.scan_unroll)
        self.embedding = nn.Embed(c.vocab, c.dim, embedding_init=jinit.normal(stddev=0.02))
        self.stack = LaneStack(c.dec_layers, c.lanes, c.dim, partial(DecoderBlock, c, dt), c.sites, dtype=dt,
                               remat=c.remat, unroll=c.scan_unroll)
        self.engrams = [Engram(c.dim, len(c.orders) * c.hashes, c.slots, c.engram_sub_dim, c.dec_layers,
                               max(c.orders), dt) for _ in c.sites]
        self.pe_gate = self.param("pe_gate", jinit.zeros, ())

    def encode(self, wav, lengths):
        """16 kHz samples (B, N) and sample counts -> encoder states (B, T, dim) and valid frame counts."""
        c = self.cfg
        x, lengths = features(c, wav, lengths)
        x, lengths = self.stem(x.astype(c.jax_dtype), lengths)
        x = jnp.pad(x, ((0, 0), (0, x.shape[1] % 2), (0, 0)))
        seen = jnp.arange(x.shape[1]) < lengths[:, None]
        mask = jnp.broadcast_to(seen[:, None, None, :], (len(seen), 1, seen.shape[1], seen.shape[1]))
        rope = precompute_rope_freqs(c.qk_dim, x.shape[1], c.rope)
        return self.encoder(x, mask=mask, valid=seen[..., None], rope=rope), lengths

    def decode(self, tokens, enc, lengths):
        c = self.cfg
        pe = nn.sigmoid(self.pe_gate).astype(enc.dtype) * sinusoids(c, enc.shape[1]).astype(enc.dtype)
        mem, mem_mask = enc + pe, jnp.arange(enc.shape[1]) < lengths[:, None]
        mask = make_causal_mask(tokens.shape[1])
        rope = precompute_rope_freqs(c.qk_dim, tokens.shape[1], c.rope)
        x = self.embedding(tokens) * math.sqrt(c.dim)
        indices = engram_indices(tokens, c.orders, c.hashes, c.slots)
        ngram_ok = jnp.stack([_mask_diag(mask, o - 1) for o in c.orders for _ in range(c.hashes)], axis=-1)
        tap_ok = jnp.stack([_mask_diag(mask, j * max(c.orders)) for j in range(ENGRAM_CONV_TAPS)])
        pairs = [e(indices, ngram_ok, tap_ok) for e in self.engrams]
        kv = (jnp.stack([k for k, _ in pairs]), jnp.stack([v for _, v in pairs])) if pairs else None
        return self.stack(x, mask=mask, rope=rope, engram_kv=kv, mem=mem, mem_mask=mem_mask)

    def logits(self, tokens, enc, frames):
        x = self.decode(tokens, enc, frames)
        return x.astype(jnp.float32) @ self.embedding.embedding.T.astype(jnp.float32)

    def __call__(self, wav, lengths, tokens):
        enc, frames = self.encode(wav, lengths)
        return self.logits(tokens, enc, frames)


def read_checkpoint(path):
    """Params tree and saved config dict from a Whistle .safetensors checkpoint."""
    import json
    from safetensors import safe_open
    from safetensors.numpy import load_file

    with safe_open(str(path), framework="np") as handle:
        metadata = handle.metadata() or {}
    params = {}
    for name, value in load_file(str(path)).items():
        node = params
        *parents, leaf = name.split("/")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return params, json.loads(metadata.get("config") or "{}")


def write_checkpoint(path, params, config, **metadata):
    """Inverse of read_checkpoint: float32 tensors under "/"-joined names, `config` as JSON."""
    import json
    from flax.traverse_util import flatten_dict
    from safetensors.numpy import save_file

    # np.array, not np.ascontiguousarray, which would turn 0-d scalars such as pe_gate into shape (1,).
    flat = {"/".join(k): np.array(v, np.float32, order="C") for k, v in flatten_dict(params).items()}
    save_file(flat, str(path), metadata={"config": json.dumps(config), **{k: str(v) for k, v in metadata.items()}})


def load_whistle(path, dtype="float32"):
    params, saved = read_checkpoint(path)
    if "encoder" not in params or "stem" not in params:
        raise ValueError(f"{path} is not a Whistle checkpoint")
    return params, WhistleConfig.from_saved(saved, dtype=dtype)
