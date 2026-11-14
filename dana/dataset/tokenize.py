"""Standalone tokenization pipeline: train tokenizer and pre-tokenize all data.

Reads the tool-calling dataset from data/tool_calls_unified/ (written by
`dana generate-data`), optionally
retrains the SentencePiece tokenizer, and tokenizes train + val splits into
.data_cache/.

Usage:
    dana tokenize                         # full run
    dana tokenize --max-samples 1000      # dev/test
"""

import os
import shutil

from .dataset import (
    CACHE_DIR,
    DEFAULT_MAX_DEC_LEN,
    DEFAULT_MAX_ENC_LEN,
    LOCAL_UNIFIED_DIR,
    _cache_key,
    _save_cache_metadata,
    get_tokenizer,
    load_tool_calls,
    pack_sequences,
    prepare_tool_call_pairs,
    train_tokenizer,
)


def _ensure_dataset():
    """Make sure data/tool_calls_unified/{train,validation} exist on disk."""
    for split in ("train", "validation"):
        split_dir = os.path.join(LOCAL_UNIFIED_DIR, split)
        if os.path.isdir(split_dir) and any(f.endswith(".arrow") for f in os.listdir(split_dir)):
            continue
        raise FileNotFoundError(
            f"No '{split}' split at {split_dir}. Run 'dana generate-data' first."
        )


def tokenize(args):
    print("=== Checking the tool-call dataset ===")
    _ensure_dataset()

    if os.path.exists(CACHE_DIR):
        print(f"Removing old tokenized cache {CACHE_DIR}/ ...")
        shutil.rmtree(CACHE_DIR)

    if getattr(args, "train_tokenizer", False):
        print("\n=== Training tokenizer (the released weights will no longer match) ===")
        train_tokenizer(max_samples=args.max_samples, force=True)
    tokenizer = get_tokenizer()

    print("\n=== Tokenizing text data ===")
    max_enc_len = getattr(args, "max_enc_len", DEFAULT_MAX_ENC_LEN)
    max_dec_len = getattr(args, "max_dec_len", DEFAULT_MAX_DEC_LEN)

    for split in ("train", "val"):
        print(f"\n--- {split} split ---")
        ds, global_indices = load_tool_calls(
            split=split,
            max_samples=args.max_samples,
            return_global_indices=True,
        )
        shuffle_tools = getattr(args, "shuffle_tools", True)
        max_tool_len = getattr(args, "max_tool_len", 256)
        prepare_tool_call_pairs._max_tool_len = max_tool_len
        enc_vl, dec_in_vl, dec_tgt_vl, loss_vl, kept_indices, _ = prepare_tool_call_pairs(
            ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
            shuffle_tools=shuffle_tools,
        )
        text_cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len,
                                   1.0, 1.0, 1.0, shuffle_tools)
        cache_path = os.path.join(CACHE_DIR, text_cache_id)
        pack_sequences(cache_path, enc_vl, dec_in_vl, dec_tgt_vl, loss_vl)

        _save_cache_metadata(split, text_cache_id, len(kept_indices),
                             max_enc_len, max_dec_len, max_tool_len)

    print("\n=== Tokenization pipeline complete ===")
