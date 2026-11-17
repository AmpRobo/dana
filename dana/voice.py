"""Voice commands: Whistle hears what you say, Dana turns it into tool calls.

Each utterance from the microphone is transcribed by Whistle, as in `dana listen`. One
that starts with the wake word ("Dana, turn on the kitchen lights.") is a command: the
words after the wake word go to Dana with the tools. Saying the wake word alone makes
the next utterance, within --wait seconds, the command. Commands and their tool calls
are printed as one JSON line each on stdout, for another program to carry out; what
was heard goes to stderr.

Words from the tool definitions (tool and parameter names, the values their
descriptions quote, other description words) are favoured while Whistle decodes, and
decoding is a beam search, so "kitten lights" comes out as "kitchen lights" more often.

Usage:
    dana voice --tools examples/tools/smart_home.json
    dana voice --tools tools.json --wake ""        # every utterance is a command
    dana voice --tools tools.json --dana-checkpoint checkpoints/dana_smart_home.pkl   # after scripts/voice_finetune.py
    dana voice --tools tools.json --tool-boost 0 --beam-size 1   # plain greedy, only --keywords favoured
    dana voice --tools examples/tools/desktop.json --speak       # Kokoro says each call's reply

With --speak, Kokoro reads out what each called tool says to say (a "say" field, else the
text its description quotes, else the description; see dana/speak.py). The microphone is
muted while it speaks, so Dana does not hear itself, wake word included.
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


# Words too common or too generic to be worth favouring while decoding.
_STOPWORDS = frozenset("""
a an the of to in on at for and or is it my me i you your this that with from by as be do not no so up
out if but all just about can will am are was were been has have had would could should may might its
our we they them their he she him her his who what when where how which some any each every also than
then now here there very too more much many most other new old get set make take let use used using via
true false null none yes given whether optional specific current currently name names number numbers
value values type types e.g eg i.e ie etc like such one two default string boolean integer float list
items item id ids format either another into over only per otherwise else
""".split())


def tool_keywords(tools, limit=300):
    """Words and phrases worth favouring while Whistle decodes commands for `tools`.

    `tools` is the tools JSON (a string or the parsed list). In order of priority:
    values the descriptions quote or list as enum ("living room", "arm home"), words of
    tool and parameter names (thermostat, brightness), then other description words of
    four letters or more. Common words are left out, as Whistle hears them well and
    favouring them only invites wrong insertions. At most `limit`, without repeats.
    """
    if isinstance(tools, str):
        tools = json.loads(tools)
    quoted, named, described = [], [], []

    def words(text):
        return re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]|[A-Za-z]", text)

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        texts = [tool.get("description", "")]
        named += tool.get("name", "").replace("-", "_").split("_")
        params = tool.get("parameters") or {}
        if isinstance(params, dict) and isinstance(params.get("properties"), dict):
            params = params["properties"]  # JSON-Schema style tools
        for key, spec in (params.items() if isinstance(params, dict) else []):
            named += key.replace("-", "_").split("_")
            if not isinstance(spec, dict):
                continue
            texts.append(spec.get("description", ""))
            quoted += [str(v) for v in spec.get("enum", []) if isinstance(v, str)]
        for text in texts:
            quoted += re.findall(r"'([^']{1,40})'", text or "") + re.findall(r'"([^"]{1,40})"', text or "")
            described += words(text or "")

    found, seen = [], set()
    for group in (quoted, named, described):
        for phrase in group:
            phrase = " ".join(phrase.replace("_", " ").split())
            if not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", phrase):
                continue  # "~/code", "ctrl+c", "101.1 FM": not written as spoken
            # one word: skip common, short and camel/snake identifiers' leftovers
            if " " not in phrase and (phrase.lower() in _STOPWORDS or len(phrase) < 3):
                continue
            if group is described and len(phrase) < 4:
                continue
            if group is not quoted:
                phrase = phrase.lower()
            if phrase.lower() in seen:
                continue
            seen.add(phrase.lower())
            found.append(phrase)
            if len(found) >= limit:
                return found
    return found


def whistle_keywords(args, tools=None):
    """--keywords (the wake word by default) at --boost, then the tools' words at --tool-boost.

    `tools` defaults to --tools when given (a file or the JSON); with neither, only --keywords.
    Prints to stderr which tool words are favoured."""
    from .whistle.transcribe import split_keywords

    mine = split_keywords(args.keywords)
    keywords = list(mine)
    if tools is None and getattr(args, "tools", None):
        tools = read_tools(args.tools)
    if tools is not None and getattr(args, "tool_boost", 0) > 0:
        taken = {k.lower() for k in mine}
        keywords += [(w, args.tool_boost) for w in tool_keywords(tools, args.max_tool_keywords)
                     if w.lower() not in taken]
        favoured = [k[0] for k in keywords if not isinstance(k, str)]
        if favoured:
            shown = ", ".join(favoured[:12]) + (", ..." if len(favoured) > 12 else "")
            print(f"Favouring {len(favoured)} words from the tools (boost {args.tool_boost:g}): {shown}",
                  file=sys.stderr)
    return keywords


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
    from .speak import Speaker, reply_for, strip_say
    from .whistle.listen import Gate, heard, load_transcriber

    defined = read_tools(args.tools)
    tools = strip_say(defined)  # "say" is for --speak, not for Dana to read
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
    speaker, gate = None, Gate(args.mute_tail)
    if args.speak:
        print(f"Loading Kokoro ({args.speak_voice})...", file=sys.stderr, flush=True)
        speaker = Speaker.from_args(args)
        speaker.warmup()
    transcriber = load_transcriber(args)
    keywords = whistle_keywords(args, tools)
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
        for result, _, spent in heard(args, transcriber, keywords, gate):
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
            reply = reply_for(found, json.loads(defined)) if speaker else ""
            if reply:
                print(f"  saying  {reply}", file=sys.stderr, flush=True)
                with gate.closed():  # the microphone hears nothing until the reply has played
                    speaker.say(reply)
    except KeyboardInterrupt:
        pass
