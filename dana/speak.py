"""Spoken replies for `dana voice --speak`: Kokoro reads out what the called tool says to say.

What a call says comes from its tool definition, in this order:
  - a "say" field, with {parameter} filled from the call's arguments:
        {"name": "set_lights", "say": "Turning the {room} lights {state}.", ...}
    "say": "" keeps a tool quiet. The field is taken out before the tools go to Dana,
    so adding it does not change what Dana reads.
  - text the description quotes in double quotes, as introduce_yourself does:
        "description": "Introduce yourself: \"Hi! I'm Dana, ...\""
  - otherwise the description itself, as a confirmation ("Minimize the current window.").

Audio is synthesised one sentence at a time and piped to `aplay`, so playback starts after
the first sentence rather than the whole reply.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

DEFAULT_MODEL = os.path.join("checkpoints", "kokoro", "kokoro-v1.0.onnx")
DEFAULT_VOICES = os.path.join("checkpoints", "kokoro", "voices-v1.0.bin")
SENTENCE_GAP = 0.15  # seconds of silence between sentences


def strip_say(tools_json):
    """`tools_json` without the "say" fields, compact, for Dana to read."""
    tools = json.loads(tools_json)
    for tool in tools:
        if isinstance(tool, dict):
            tool.pop("say", None)
    return json.dumps(tools, separators=(",", ":"))


def _fill(template, arguments):
    def value(match):
        v = arguments.get(match.group(1), "")
        return ("on" if v else "off") if isinstance(v, bool) else str(v)
    return re.sub(r"\{(\w+)\}", value, template)


def reply_for(calls, tools):
    """What to say for Dana's `calls` (a list of {"name", "arguments"}), given the tools list.

    Calls to tools not in `tools`, and output that is not a list of calls, say nothing."""
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return ""
    by_name = {t.get("name"): t for t in tools if isinstance(t, dict)}
    parts = []
    for call in calls:
        tool = by_name.get(call.get("name")) if isinstance(call, dict) else None
        if tool is None:
            continue
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if "say" in tool:
            text = _fill(str(tool["say"] or ""), arguments)
        else:
            description = str(tool.get("description") or "")
            quoted = re.findall(r'"([^"]+)"', description)
            text = " ".join(quoted) if quoted else description
        if text.strip():
            parts.append(text.strip())
    return " ".join(parts)


class Speaker:
    """Kokoro voice played through `aplay`."""

    def __init__(self, model=DEFAULT_MODEL, voices=DEFAULT_VOICES, voice="af_heart", speed=1.0, lang=None,
                 device=None):
        if not shutil.which("aplay"):
            raise SystemExit("--speak needs aplay (install alsa-utils)")
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise SystemExit('--speak needs Kokoro: pip install -e ".[voice]" (or run ./setup)') from None
        for path in (model, voices):
            if not os.path.exists(path):
                raise SystemExit(f"--speak needs {path} (./setup downloads it, from model-files-v1.0 of "
                                 "github.com/thewh1teagle/kokoro-onnx)")
        self.kokoro = Kokoro(model, voices)
        if voice not in self.kokoro.get_voices():
            raise SystemExit(f"--speak-voice must be one of {', '.join(sorted(self.kokoro.get_voices()))}")
        self.voice, self.speed, self.device = voice, speed, device
        self.lang = lang or ("en-gb" if voice.startswith("b") else "en-us")

    @classmethod
    def from_args(cls, args):
        return cls(args.kokoro_model, args.kokoro_voices, args.speak_voice, args.speak_speed, args.speak_lang,
                   args.speak_device)

    def _synth(self, text):
        audio, rate = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        return (np.clip(audio, -1, 1) * 32767).astype("<i2"), rate

    def warmup(self):
        self._synth("Hello.")

    def say(self, text):
        """Speak `text`, returning once playback has finished."""
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        player, done = None, False
        try:
            for sentence in sentences:
                pcm, rate = self._synth(sentence)
                if player is None:
                    cmd = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1"]
                    if self.device:
                        cmd += ["-D", self.device]
                    player = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                gap = np.zeros(int(rate * SENTENCE_GAP), "<i2")
                player.stdin.write(pcm.tobytes() + gap.tobytes())
            done = True
        except BrokenPipeError:
            print("  aplay stopped before the reply was spoken", file=sys.stderr, flush=True)
        finally:
            if player is not None:
                if done:
                    try:
                        player.stdin.close()
                    except BrokenPipeError:
                        pass
                    player.wait()  # until the last sentence has played
                else:
                    player.kill()
                    player.wait()
