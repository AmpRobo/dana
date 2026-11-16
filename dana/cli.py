import argparse
import os
import re
import sys
import threading

HELP = """Check the readme"""

_GOOGLE_PCI_VENDOR = "0x1ae0"


def _is_tpu_host():
    """Whether to call jax.distributed.initialize(): DANA_DISTRIBUTED=1/0 decides, else a
    Google /dev/accel0. Other vendors' accel devices (an Intel NPU, say) are not TPUs."""
    forced = os.environ.get("DANA_DISTRIBUTED")
    if forced is not None:
        return forced == "1"
    if not os.path.exists("/dev/accel0"):
        return False
    try:
        with open("/sys/class/accel/accel0/device/vendor") as f:
            return f.read().strip() == _GOOGLE_PCI_VENDOR
    except OSError:
        return True


_ABSL_LOG_START = re.compile(rb"^[EIWF]\d{4} \d\d:\d\d:\d\d")
_NOISY_LOG_HEADER = re.compile(
    rb"\] (?:Fusion: .*gemm_fusion|Computation: .*_computation|Delay kernel timed out)"
)

_log_filter_installed = False


def _install_xla_log_filter():
    """Drop XLA Triton autotuner noise from stderr.

    XLA's Triton GEMM autotuner logs failed candidate fusions via LOG(ERROR)
    in xtile_compiler.cc and cuda_timer.cc. These are unconditional and do
    not respect TF_CPP_MIN_LOG_LEVEL, so we filter them at the file
    descriptor level.

    Strategy:
      - Rebind Python's sys.stderr to a fresh file object over the real
        terminal fd, so tqdm and print() writes go straight to the terminal
        and never enter our pipe. This keeps progress bars (which use \\r
        without trailing \\n) from stalling the filter's line parser.
      - Replace fd 2 with a pipe. Only C-level writes (absl / XLA LOG(...))
        now flow through the pipe, and they are always \\n-terminated and
        well-formed, so a simple line-based filter is reliable.
    """
    global _log_filter_installed
    if _log_filter_installed:
        return
    _log_filter_installed = True

    # 1. Dup the original stderr fd; rebind Python's sys.stderr to it so
    #    all Python-level writes bypass the pipe.
    py_stderr_fd = os.dup(2)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    sys.stderr = os.fdopen(py_stderr_fd, "w", encoding="utf-8",
                           errors="replace", buffering=1)

    # 2. Dup another copy for the filter thread to write filtered C-level
    #    output to.
    out_fd = os.dup(2)  # before dup2 below, fd 2 is still the real terminal

    # 3. Redirect fd 2 to the write end of a new pipe. All C-level writes
    #    from XLA now land in the pipe.
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def pump():
        reader = os.fdopen(r_fd, "rb", buffering=0)
        out = os.fdopen(out_fd, "wb", buffering=0)
        buf = b""
        skipping = False
        try:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    idx = buf.find(b"\n")
                    if idx == -1:
                        break
                    line = bytes(buf[:idx])
                    buf = buf[idx + 1:]
                    is_log_start = bool(_ABSL_LOG_START.match(line))
                    if skipping:
                        if is_log_start:
                            if _NOISY_LOG_HEADER.search(line):
                                continue
                            skipping = False
                            out.write(line + b"\n")
                        # else: continuation body of a skipped log block — drop
                    else:
                        if is_log_start and _NOISY_LOG_HEADER.search(line):
                            skipping = True
                            continue
                        out.write(line + b"\n")
        except Exception:
            pass

    t = threading.Thread(target=pump, daemon=True, name="xla-log-filter")
    t.start()


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
_install_xla_log_filter()

from .dataset.tokenizer import DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_DEC_LEN, DEFAULT_MAX_GEN_LEN


def _add_tool_bias_args(p, required=False):
    """Options shared by `transcribe`, `listen` and `voice`: favouring words from tool definitions."""
    p.add_argument("--tools", type=str, required=required, default=None,
                   help="Tools JSON: a file such as examples/tools/smart_home.json, or the JSON itself"
                        + ("" if required else "; its words are favoured while decoding"))
    p.add_argument("--tool-boost", type=float, default=2.0,
                   help="Logit bonus for words taken from --tools (names, quoted values, descriptions); "
                        "0 turns it off (default: 2.0)")
    p.add_argument("--max-tool-keywords", type=int, default=300,
                   help="At most this many words from the tools are favoured (default: 300)")
    p.add_argument("--beam-size", type=int, default=4,
                   help="Whistle beam search width; 1 is greedy (default: 4)")


def _add_microphone_args(p, language):
    """Options shared by `listen` and `voice`: Whistle weights, the microphone and keyword biasing."""
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Whistle weights (default: checkpoints/whistle_dana.safetensors if present, "
                        "else checkpoints/whistle.safetensors)")
    p.add_argument("--language", type=str, default=language,
                   help=f"en, de, fr, es, it, nl or pl (default: {language or 'detect per utterance'})")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--device", type=str, default=None, help="arecord capture device, e.g. hw:0,6 (default: system)")
    p.add_argument("--stdin", action="store_true", help="Read raw 16 kHz S16_LE mono audio from stdin")
    p.add_argument("--silence", type=float, default=0.8, help="Seconds of quiet that end an utterance")
    p.add_argument("--threshold", type=float, default=None, help="Speech RMS threshold (default: calibrate)")
    p.add_argument("--calibrate", type=float, default=1.0, help="Seconds of room noise to calibrate on")
    p.add_argument("--keywords", type=str, default="Dana",
                   help='Words or phrases to favour, comma separated, spelled as they should appear '
                        '(default: Dana; "" for none)')
    p.add_argument("--boost", type=float, default=3.0, help="How strongly to favour --keywords (logit bonus)")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="dana", add_help=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train", add_help=False)
    p.add_argument("--name", type=str, default="baseline",
                   help="Experiment name for checkpoints and wandb (default: baseline)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Full resume: adopts checkpoint's config and step counter")
    p.add_argument("--init-from", type=str, default=None,
                   help="Initialize params from a local pretrained base "
                        "(e.g. dana_base.pkl). Uses CLI config; partial-loads "
                        "matching params, random-inits the rest.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-dec-layers", type=int, default=8)
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--decay-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--max-eval-samples", type=int, default=5000)
    p.add_argument("--contrastive-weight", type=float, default=0.1,
                   help="Weight for CLIP-style contrastive loss (default: 0.1)")
    p.add_argument("--contrastive-dim", type=int, default=128,
                   help="Dimension of contrastive projection head (default: 128)")
    p.add_argument("--w-name", type=float, default=2.0,
                   help="Loss weight for tool name tokens (default: 2.0)")
    p.add_argument("--w-value", type=float, default=4.0,
                   help="Loss weight for argument value tokens (default: 4.0)")
    p.add_argument("--w-key", type=float, default=1.5,
                   help="Loss weight for argument key tokens (default: 1.5)")

    p = sub.add_parser("pretrain", add_help=False)
    p.add_argument("--name", type=str, default="pretrain",
                   help="Experiment name for wandb (default: pretrain)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume from checkpoint (e.g. checkpoints/dana_base.pkl)")
    p.add_argument("--resume-step", type=int, default=None,
                   help="Override resume step (skip this many batches)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-dec-layers", type=int, default=8)
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--decay-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after N steps (default: full epoch)")
    p.add_argument("--save-every", type=int, default=1000,
                   help="Save checkpoint every N steps (default: 1000)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--data", type=str, required=True,
                   help="Local .parquet files (path, directory or glob)")

    p = sub.add_parser("tokenize", add_help=False)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples per split (for dev/test)")
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN,
                   help=f"Max encoder sequence length (default: {DEFAULT_MAX_ENC_LEN})")
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN,
                   help=f"Max decoder sequence length (default: {DEFAULT_MAX_DEC_LEN})")
    p.add_argument("--shuffle-tools", action=argparse.BooleanOptionalAction, default=True,
                   help="Shuffle tool order in encoder input (default: True)")
    p.add_argument("--max-tool-len", type=int, default=256,
                   help="Max token length for individual tool descriptions (default: 256)")
    p.add_argument("--train-tokenizer", action="store_true",
                   help="Retrain the SentencePiece tokenizer on the data first (breaks compatibility "
                        "with the released weights; only for pretraining from scratch)")

    p = sub.add_parser("run", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--query", type=str, default=None, help="Query text for tool-call generation")
    p.add_argument("--tools", type=str, default=None, help="Tools JSON for tool-call generation")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-constrained", action="store_true",
                   help="Disable grammar-constrained decoding for tool names/arg keys")

    p = sub.add_parser("transcribe", add_help=False)
    p.add_argument("audio", nargs="+", help="WAV file(s), any sample rate")
    p.add_argument("--checkpoint", type=str, default="checkpoints/whistle.safetensors")
    p.add_argument("--language", type=str, default=None,
                   help="en, de, fr, es, it, nl or pl (default: detect)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--json", action="store_true", help="Print one JSON object per file")
    p.add_argument("--keywords", type=str, default=None,
                   help="Words or phrases to favour, comma separated, spelled as they should appear")
    p.add_argument("--boost", type=float, default=3.0, help="How strongly to favour --keywords (logit bonus)")
    _add_tool_bias_args(p)

    p = sub.add_parser("listen", add_help=False)
    _add_microphone_args(p, language=None)
    _add_tool_bias_args(p)

    p = sub.add_parser("voice", add_help=False)
    p.add_argument("--dana-checkpoint", type=str, default="checkpoints/dana.pkl")
    p.add_argument("--wake", type=str, default="Dana",
                   help='Wake word a command starts with (default: Dana; "" makes every utterance a command)')
    p.add_argument("--wait", type=float, default=10.0,
                   help="Seconds after a lone wake word in which the next utterance is the command")
    _add_microphone_args(p, language="en")
    _add_tool_bias_args(p, required=True)

    p = sub.add_parser("record", add_help=False)
    p.add_argument("text", type=str, help="Transcript of what you will say, e.g. \"Dana.\"")
    p.add_argument("--count", type=int, default=5, help="How many clips to record")
    p.add_argument("--out", type=str, default="data/whistle/clips", help="Folder for the WAVs and clips.jsonl")
    p.add_argument("--language", type=str, default="en")
    p.add_argument("--device", type=str, default=None, help="arecord capture device, e.g. hw:0,6 (default: system)")
    p.add_argument("--silence", type=float, default=0.8, help="Seconds of quiet that end a clip")
    p.add_argument("--threshold", type=float, default=None, help="Speech RMS threshold (default: calibrate)")
    p.add_argument("--calibrate", type=float, default=1.0, help="Seconds of room noise to calibrate on")

    p = sub.add_parser("finetune-whistle", add_help=False)
    p.add_argument("train", type=str, help='JSONL of {"audio": WAV path, "text": transcript, "language": code}')
    p.add_argument("--validation", type=str, default=None, help="JSONL scored after every epoch")
    p.add_argument("--checkpoint", type=str, default="checkpoints/whistle.safetensors", help="Base checkpoint")
    p.add_argument("--out", type=str, default="checkpoints/whistle_tuned.safetensors",
                   help="Merged checkpoint to write")
    # 5 epochs at 3e-4: cactus's 3 at 1e-4 left a few hundred clips of one new word unlearnt.
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--decoder-only", action="store_true", help="Adapt only the decoder's attention")
    p.add_argument("--seconds", type=int, default=None,
                   help="Pad every clip to this many seconds (default: the longest clip, rounded up)")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("eval", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-eval-samples", type=int, default=5000)
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    p.add_argument("--max-gen-len", type=int, default=DEFAULT_MAX_GEN_LEN)
    p.add_argument("--tool-call-samples", type=int, default=200,
                   help="Samples for tool-call accuracy eval (default: 200)")
    p.add_argument("--throughput-runs", type=int, default=10)
    p.add_argument("--no-constrained", action="store_true",
                   help="Disable grammar-constrained decoding for tool names/arg keys")

    p = sub.add_parser("generate-data", add_help=False)
    p.add_argument("--num-samples", type=int, default=500, help="Number of samples to generate")
    p.add_argument("--batch-size", type=int, default=25, help="Examples per Gemini call")
    p.add_argument("--workers", type=int, default=8, help="Parallel Gemini calls")
    p.add_argument("--model", type=str, default=None, help="Gemini model override")
    p.add_argument("--dry-run", action="store_true", help="Generate only, skip save")
    p.add_argument("--output-jsonl", type=str, default=None, help="Also save raw generations to JSONL")
    p.add_argument("--save-every", type=int, default=None, help="Merge+save every N samples")

    p = sub.add_parser("evaluate", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--benchmarks", type=str, nargs="*",
                   choices=["wikitext2", "lambada", "hellaswag", "arc_easy"])
    p.add_argument("--max-samples", type=int, default=500)

    p = sub.add_parser("finetune", add_help=False)
    p.add_argument("jsonl_path", type=str, help="Path to JSONL training data")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Base model checkpoint (default: checkpoints/dana.pkl)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--max-enc-len", type=int, default=None)
    p.add_argument("--max-dec-len", type=int, default=None)

    p = sub.add_parser("playground", add_help=False)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")

    p = sub.add_parser("tpu", add_help=False)
    tpu_sub = p.add_subparsers(dest="tpu_action")

    tp = tpu_sub.add_parser("create", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--type", dest="accel_type", type=str, default="v6e-8")
    tp.add_argument("--version", type=str, default=None,
                    help="Software version (auto-detected from --type if omitted)")
    tp.add_argument("--preemptible", action="store_true", default=False,
                    help="Create a preemptible (spot) TPU VM")

    tp = tpu_sub.add_parser("connect", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("setup", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("sync", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("train", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)
    tp.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="Extra args passed to dana train")

    tp = tpu_sub.add_parser("pretrain", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)
    tp.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="Extra args passed to dana pretrain")

    tp = tpu_sub.add_parser("claude", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("stop", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("start", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("delete", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tpu_sub.add_parser("list", add_help=False)

    args = parser.parse_args()

    if not args.command:
        print(HELP)
        sys.exit(0)

    if args.command == "tokenize":
        from .dataset.tokenize import tokenize
        tokenize(args)
    elif args.command == "pretrain":
        import jax
        if _is_tpu_host():
            jax.distributed.initialize()
        from .training.pretrain import pretrain
        pretrain(args)
    elif args.command == "train":
        import jax
        if _is_tpu_host():
            jax.distributed.initialize()
        from .training.train import train
        train(args)
    elif args.command == "run":
        from .model.run import main as run_main
        run_main(args)
    elif args.command == "transcribe":
        from .whistle.transcribe import main as transcribe_main
        transcribe_main(args)
    elif args.command == "listen":
        from .whistle.listen import main as listen_main
        listen_main(args)
    elif args.command == "voice":
        from .voice import main as voice_main
        voice_main(args)
    elif args.command == "record":
        from .whistle.record import main as record_main
        record_main(args)
    elif args.command == "finetune-whistle":
        from .whistle.finetune import finetune_main
        finetune_main(args)
    elif args.command == "eval":
        from .training.eval import main as eval_main_fn
        eval_main_fn(args)
    elif args.command == "generate-data":
        from .dataset.generate import main as gendata_main, MODEL as _MODEL, SAVE_EVERY as _SE
        if args.model is None:
            args.model = _MODEL
        if args.save_every is None:
            args.save_every = _SE
        gendata_main(args)
    elif args.command == "finetune":
        from .training.finetune import finetune_local
        finetune_local(args)
    elif args.command == "playground":
        from .ui.server import main as ui_main
        ui_main(args)
    elif args.command == "tpu":
        from .utils.tpu import tpu_dispatch
        tpu_dispatch(args)
