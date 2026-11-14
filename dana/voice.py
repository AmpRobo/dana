"""Voice commands: Whistle hears what you say, Dana turns it into tool calls.

Each utterance from the microphone is transcribed by Whistle, as in `dana listen`. One
that starts with the wake word ("Dana, turn on the kitchen lights.") is a command: the
words after the wake word go to Dana with the tools. Saying the wake word alone makes
the next utterance, within --wait seconds, the command. Commands and their tool calls
are printed as one JSON line each on stdout, for another program to carry out; what
was heard goes to stderr.

Usage:
    dana voice --tools examples/tools/smart_home.json
    dana voice --tools tools.json --wake ""        # every utterance is a command
"""

import json
import os
import re
import sys
import time


def _words(text):
    return [re.sub(r"\W", "", w).lower() for w in text.split()]


def after_wake(text, wake):
    """The command in `text` after the wake phrase it starts with: "" if nothing follows it,
    None if it does not start with it. With no wake phrase all of `text` is the command."""
    if not wake:
        return text.strip()
    wanted, words = _words(wake), text.split()
    if _words(" ".join(words[:len(wanted)])) != wanted:
        return None
    rest = " ".join(words[len(wanted):]).lstrip(" ,.;:!?-")
    return rest[:1].upper() + rest[1:]


def read_tools(value):
    """The tools JSON list from a file path or a JSON string, re-serialised compactly."""
    text = open(value, encoding="utf-8").read() if os.path.isfile(value) else value
    try:
        tools = json.loads(text)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--tools is neither a JSON file nor JSON: {error}") from None
    if not isinstance(tools, list):
        raise SystemExit("--tools must be a JSON list of tool definitions")
    return json.dumps(tools, separators=(",", ":"))


def main(args):
    from .dataset.dataset import DEFAULT_RUN_ENC_LEN, get_tokenizer
    from .model.architecture import SimpleAttentionNetwork
    from .model.run import generate, load_checkpoint, normalize_tools
    from .whistle.listen import heard, load_transcriber

    tools = read_tools(args.tools)
    print(f"Loading Dana ({args.dana_checkpoint})...", file=sys.stderr, flush=True)
    params, config = load_checkpoint(args.dana_checkpoint)
    model, tokenizer = SimpleAttentionNetwork(config), get_tokenizer()

    def calls(command):
        raw = generate(model, params, tokenizer, command, tools=tools, stream=False)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw  # not valid JSON: pass Dana's text on as it is

    calls("Hello.")  # compile the decoder now rather than on the first command
    transcriber = load_transcriber(args)
    names = ", ".join(t["name"] for t in json.loads(tools))
    size = len(tokenizer.encode(normalize_tools(tools)[0]))
    print(f"Tools ({size} tokens): {names}", file=sys.stderr)
    if size > DEFAULT_RUN_ENC_LEN - 64:  # leave room for the command
        print(f"Warning: Dana reads at most {DEFAULT_RUN_ENC_LEN} tokens, so the last tools are cut off",
              file=sys.stderr)
    print(f'Say "{args.wake}, ..." to give a command.' if args.wake else "Every utterance is a command.",
          file=sys.stderr, flush=True)

    armed = None  # when the wake word was last said on its own
    try:
        for result, _, spent in heard(args, transcriber):
            text = result["text"]
            command = after_wake(text, args.wake)
            if command is None and armed is not None and time.time() - armed <= args.wait:
                command = text
            armed = None
            if command is None:
                print(f"  heard   {text}", file=sys.stderr, flush=True)
                continue
            if not command:
                armed = time.time()
                print(f"  heard   {text}  (waiting {args.wait:g} s for the command)", file=sys.stderr, flush=True)
                continue
            start = time.time()
            found = calls(command)
            print(f"  command {command}  ({spent:.1f} s to hear, {time.time() - start:.1f} s to call)",
                  file=sys.stderr, flush=True)
            print(json.dumps({"command": command, "calls": found}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass
