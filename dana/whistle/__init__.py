"""Whistle - speech-to-text in seven languages, standalone from Dana's text model.

Ported from cactus-compute/cactus @ bafcf73 (Apache-2.0, see LICENSE-APACHE):
model structure and parameter names are unchanged so the official checkpoint
loads as is, but nothing here imports Dana's or Needle 3's model code.
"""

from .model import LANGUAGES, Whistle, WhistleConfig, load_whistle  # noqa: F401
from .transcribe import Transcriber, WhistleTokenizer, read_wav  # noqa: F401
