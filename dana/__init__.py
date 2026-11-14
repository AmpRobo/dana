"""Dana - a 26M parameter function calling model."""

import os as _os

# Everything is local; keep the datasets library off the Hugging Face Hub.
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"
_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from dana.model.architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
)
from dana.model.run import (
    generate,
    generate_batch,
    load_checkpoint,
    encode_for_retrieval,
    retrieve_tools,
)
from dana.dataset.dataset import get_tokenizer

__all__ = [
    "SimpleAttentionNetwork",
    "TransformerConfig",
    "generate",
    "generate_batch",
    "load_checkpoint",
    "encode_for_retrieval",
    "retrieve_tools",
    "get_tokenizer",
]
