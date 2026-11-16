"""From a tools JSON file to a Dana that copes with Whistle's mistakes, in one command.

    python scripts/voice_finetune.py examples/tools/smart_home.json

Stages (each writes into --out and is skipped next time if its output is there, so an
interrupted run picks up where it stopped; --redo STAGE runs one again):

  1. generate  ~150 spoken-style commands per tool with their tool calls, written by Gemini
               (needs GEMINI_API_KEY), or your own JSONL via --data. -> examples.jsonl
  2. speak     each command read aloud by a local TTS (Kokoro, piper, espeak-ng, espeak
               or pico2wave), with random noise, volume and speed, then transcribed by Whistle
               with the same settings as `dana voice` (beam search, tool words favoured).
               What Whistle wrote is kept, mistakes and all. -> heard.jsonl
  3. build     training rows: the clean command, what Whistle heard, and extra copies with
               the kinds of mistakes Whistle made (learnt from stage 2) applied to other
               commands, all with the correct tool calls. Every version of a command stays
               in the same train/val/test split. -> train.jsonl
  4. train     `dana finetune` on train.jsonl. -> checkpoints/dana_<name>.pkl
  5. eval      base vs finetuned Dana on the held-out commands, clean and as heard.
               -> report.json

Then:  dana voice --tools examples/tools/smart_home.json --dana-checkpoint checkpoints/dana_smart_home.pkl

Examples:
    python scripts/voice_finetune.py tools.json --data my_examples.jsonl          # no Gemini
    python scripts/voice_finetune.py tools.json --tts kokoro      # checkpoints/kokoro/kokoro-v1.0.onnx + voices
    python scripts/voice_finetune.py tools.json --tts piper --piper-model en_US-lessac-medium.onnx
    python scripts/voice_finetune.py tools.json --stages generate,speak,build     # data only
    python scripts/voice_finetune.py tools.json --redo build --text-copies 3      # rebuild, more noise

--data rows are {"query": ..., "answers": [tool calls] or a JSON string of them}, with an
optional "split" ("train"/"val"/"test"); any "tools" field is replaced by the tools file.
Without a TTS on the machine, stage 2 is skipped and stage 3 adds generic recognition-style
noise only; `sudo apt install espeak-ng` (or piper, which sounds more human) is better.
"""
import argparse
import collections
import difflib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

STAGES = ("generate", "speak", "build", "train", "eval")
BATCH = 25  # commands per Gemini call


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- tools and calls

def load_tools(path):
    with open(path, encoding="utf-8") as f:
        tools = json.load(f)
    if not isinstance(tools, list) or not all(isinstance(t, dict) and t.get("name") for t in tools):
        raise SystemExit(f"{path}: expected a JSON list of tools, each with a name")
    return tools


def tool_params(tool):
    """{param: spec} and the set of required params, for Dana-style or JSON-Schema-style tools."""
    params = tool.get("parameters") or {}
    if isinstance(params.get("properties"), dict):
        required = set(params.get("required") or [])
        props = params["properties"]
        required |= {k for k, v in props.items() if isinstance(v, dict) and v.get("required") is True}
        return props, required
    return params, {k for k, v in params.items() if isinstance(v, dict) and v.get("required")}


_TYPES = {"string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,),
          "array": (list,), "object": (dict,)}


def valid_calls(calls, by_name):
    """Whether `calls` is a non-empty list of well-formed calls to tools in `by_name`."""
    if not isinstance(calls, list) or not calls:
        return False
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in by_name:
            return False
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            return False
        props, required = tool_params(by_name[call["name"]])
        if not required <= set(args) or not set(args) <= set(props):
            return False
        for key, value in args.items():
            kind = props[key].get("type") if isinstance(props[key], dict) else None
            ok = _TYPES.get(kind)
            if ok and (not isinstance(value, ok) or (kind in ("number", "integer") and isinstance(value, bool))):
                return False
    return True


def norm_text(text):
    return " ".join(re.sub(r"[^\w' ]", " ", text.lower()).split())


def primary(calls):
    return calls[0]["name"] if calls else "__none__"


# ---------------------------------------------------------------- 1. generate

PROMPT = """You are writing training data for a small on-device voice assistant that turns spoken \
commands into tool calls.

ALL TOOLS IT HAS:
{tools}

TASK: write {n} different things a person might SAY OUT LOUD to the assistant, {what}, each with \
the exact tool calls the assistant should make.

RULES:
- Spoken style: how people really talk to a voice assistant. Mix terse ("kitchen lights off"), \
casual, polite ("could you..."), indirect ("it's freezing in here") and longer rambling requests. \
Lengths from 2 to about 20 words.
- Write them as a speech recognizer would print them: normal capitalisation and punctuation, \
numbers mostly as digits, no emoji, no markdown, no URLs or file paths unless people say them.
- Argument values must come from what the person said (an indirect request may imply a value, \
e.g. "it's freezing" -> action "on" for a heater). Do not invent names or details they did not say.
- Use correct JSON types: numbers as numbers, booleans as true/false. Include every required \
parameter; include optional ones only when the person said them.
- Vary the values widely: different rooms, names, devices, amounts, times and wordings. Use the \
example values in the descriptions sometimes, but mostly others that fit.
- Every command must be different; do not reuse a sentence pattern.

Return ONLY a JSON array, no explanation:
[{{"query": "...", "answers": [{{"name": "tool_name", "arguments": {{"param": "value"}}}}]}}]"""


def _ask_gemini(pool, model, prompt, temperature):
    client = pool.get()
    response = client.models.generate_content(
        model=model, contents=prompt, config={"temperature": temperature, "max_output_tokens": 16384})
    text = (response.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("not a JSON list")
    return rows


def generate(tools, args):
    from dana.dataset.generate import MODEL, ClientPool, make_clients

    model = args.gemini_model or MODEL
    pool = ClientPool(make_clients())
    by_name = {t["name"]: t for t in tools}
    tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    jobs = []
    for tool in tools:
        for start in range(0, args.per_tool, BATCH):
            jobs.append((tool["name"], min(BATCH, args.per_tool - start)))
    multi = round(args.per_tool * len(tools) * args.multi) if len(tools) > 1 else 0
    for start in range(0, multi, BATCH):
        jobs.append((None, min(BATCH, multi - start)))
    log(f"Asking {model} for {sum(n for _, n in jobs)} commands in {len(jobs)} calls...")

    def run(job, attempt_seed):
        name, n = job
        what = (f'each of which calls the tool "{name}" (and only that tool)' if name else
                "each asking for two or three things at once, using different tools")
        prompt = PROMPT.format(tools=tools_json, n=n, what=what)
        rng = random.Random(attempt_seed)
        for attempt in range(4):
            try:
                rows = _ask_gemini(pool, model, prompt, rng.choice([0.8, 0.9, 1.0, 1.1]))
                return [r for r in rows if isinstance(r, dict)]
            except Exception as error:  # network, quota, bad JSON: retry with backoff
                log(f"  {name or 'multi'}: attempt {attempt + 1} failed ({str(error)[:120]})")
                time.sleep(2 ** attempt * 3)
        return []

    found, done = [], 0
    with ThreadPoolExecutor(args.workers) as executor:
        futures = [executor.submit(run, job, args.seed * 100003 + i) for i, job in enumerate(jobs)]
        for future in as_completed(futures):
            found += future.result()
            done += 1
            if done % 10 == 0 or done == len(jobs):
                log(f"  {done}/{len(jobs)} calls, {len(found)} commands")

    examples, seen, dropped = [], set(), 0
    for row in found:
        query, calls = row.get("query"), row.get("answers")
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except json.JSONDecodeError:
                calls = None
        if not isinstance(query, str) or not query.strip() or not valid_calls(calls, by_name):
            dropped += 1
            continue
        key = norm_text(query)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        examples.append({"query": query.strip(), "answers": calls})
    log(f"Kept {len(examples)} commands, dropped {dropped} (malformed calls or repeats)")
    return examples


def read_data(path, tools):
    by_name = {t["name"]: t for t in tools}
    examples, dropped = [], 0
    for row in read_jsonl(path):
        calls = row.get("answers")
        if isinstance(calls, str):
            calls = json.loads(calls)
        # [] is a request no tool can carry out: Dana should make no call
        if not (calls == [] or valid_calls(calls, by_name)) or not str(row.get("query", "")).strip():
            dropped += 1
            continue
        example = {"query": row["query"].strip(), "answers": calls}
        if row.get("split") in ("train", "val", "test"):
            example["split"] = row["split"]
        examples.append(example)
    if dropped:
        log(f"{path}: dropped {dropped} rows whose calls do not fit the tools")
    return examples


def assign_splits(examples, seed, per_tool=10):
    """Up to `per_tool` val and test commands per tool (fewer for small tools), rest train."""
    if all("split" in e for e in examples):
        return examples
    rng = random.Random(seed)
    buckets = collections.defaultdict(list)
    for e in examples:
        buckets[primary(e["answers"])].append(e)
    for items in buckets.values():
        rng.shuffle(items)
        held = min(per_tool, max(1, len(items) // 8)) if len(items) >= 3 else 0
        for i, e in enumerate(items):
            e["split"] = "test" if i < held else "val" if i < 2 * held else "train"
    return examples


# ---------------------------------------------------------------- 2. speak

ESPEAK_VOICES = {"en": ["en-us", "en-gb", "en-us+f3", "en-us+m3", "en-gb+f4", "en-us+m5", "en-gb-scotland",
                        "en-029", "en-us+f2", "en-gb+m4"]}
KOKORO_VOICES = {"en": ("af_", "am_", "bf_", "bm_"), "fr": ("ff_",), "es": ("ef_", "em_"), "it": ("if_", "im_")}
KOKORO_LANGS = {"en": "en-us", "fr": "fr-fr", "es": "es", "it": "it"}
PICO_LANGS = {"en": ["en-US", "en-GB"], "de": ["de-DE"], "fr": ["fr-FR"], "es": ["es-ES"], "it": ["it-IT"]}


def pick_tts(args):
    if args.tts == "none":
        return None
    order = [args.tts] if args.tts != "auto" else ["kokoro", "piper", "espeak-ng", "espeak", "pico2wave"]
    for engine in order:
        if engine == "kokoro":
            try:
                import kokoro_onnx  # noqa: F401
                ready = os.path.exists(args.kokoro_model) and os.path.exists(args.kokoro_voices)
            except ImportError:
                ready = False
            if ready:
                return "kokoro"
            if args.tts == "kokoro":
                raise SystemExit(f"--tts kokoro needs `pip install kokoro-onnx` and {args.kokoro_model} and "
                                 f"{args.kokoro_voices} (model-files-v1.0 of github.com/thewh1teagle/kokoro-onnx)")
            continue
        if engine == "piper":
            if args.piper_model and shutil.which(args.piper_cmd):
                return "piper"
            if args.tts == "piper":
                raise SystemExit("--tts piper needs --piper-model and the piper command (pip install piper-tts)")
            continue
        if shutil.which(engine):
            return engine
        if args.tts != "auto":
            raise SystemExit(f"--tts {engine}: command not found")
    return None


def synthesize(engine, text, path, rng, args):
    """Read `text` aloud into the WAV `path` with a random voice, speed and pitch."""
    if engine == "kokoro":
        kokoro = getattr(args, "_kokoro", None)
        if kokoro is None:
            from kokoro_onnx import Kokoro
            kokoro = args._kokoro = Kokoro(args.kokoro_model, args.kokoro_voices)
            prefixes = KOKORO_VOICES.get(args.language, ())
            args._kokoro_voices = [v for v in kokoro.get_voices() if v.startswith(prefixes)] or kokoro.get_voices()
        voices = args._kokoro_voices
        voice = rng.choice(voices)
        if rng.random() < 0.3:  # a blend of two voices: a speaker none of the presets is
            a, b = kokoro.get_voice_style(voice), kokoro.get_voice_style(rng.choice(voices))
            mix = rng.uniform(0.3, 0.7)
            voice = a * mix + b * (1 - mix)
        lang = "en-gb" if isinstance(voice, str) and voice[:1] == "b" else KOKORO_LANGS.get(args.language, "en-us")
        audio, rate = kokoro.create(text, voice=voice, speed=rng.uniform(0.85, 1.2), lang=lang)
        import soundfile
        soundfile.write(path, audio, rate, subtype="PCM_16")
    elif engine == "piper":
        cmd = [args.piper_cmd, "--model", rng.choice(args.piper_model), "--output_file", path,
               "--length_scale", f"{rng.uniform(0.85, 1.2):.2f}", "--noise_scale", f"{rng.uniform(0.5, 0.9):.2f}"]
        subprocess.run(cmd, input=text.encode(), check=True, capture_output=True)
    elif engine in ("espeak-ng", "espeak"):
        voices = ESPEAK_VOICES.get(args.language, [args.language])
        speed, pitch = str(rng.randint(135, 190)), str(rng.randint(30, 70))
        for voice in (rng.choice(voices), args.language):
            done = subprocess.run([engine, "-v", voice, "-s", speed, "-p", pitch, "-w", path, "--stdin"],
                                  input=text.encode(), capture_output=True)
            if done.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1000:
                return
        raise RuntimeError(done.stderr.decode(errors="replace")[:200])
    elif engine == "pico2wave":
        lang = rng.choice(PICO_LANGS.get(args.language, ["en-US"]))
        subprocess.run(["pico2wave", "-l", lang, "-w", path, text], check=True, capture_output=True)


def whistle_settings(tools, args):
    """Keywords as `dana voice` would favour them, so training sees the same kind of mistakes."""
    from dana.voice import whistle_keywords

    voice_args = argparse.Namespace(keywords="", tool_boost=args.tool_boost, max_tool_keywords=args.max_tool_keywords)
    return whistle_keywords(voice_args, json.dumps(tools, separators=(",", ":")))


def speak(examples, tools, args, out_path):
    engine = pick_tts(args)
    if engine is None:
        log("No TTS found (kokoro, piper, espeak-ng, espeak or pico2wave): skipping the speak stage. "
            "Training will use generic recognition-style noise only.")
        return False
    import numpy as np
    from whistle_keyword_data import augment
    from dana.whistle.listen import DEFAULT_CHECKPOINT, TUNED_CHECKPOINT
    from dana.whistle.transcribe import Transcriber, read_wav

    done = set()
    if os.path.exists(out_path):
        done = {(r["index"], r["copy"]) for r in read_jsonl(out_path)}
    todo = [(i, c) for i, _ in enumerate(examples) for c in range(args.tts_copies) if (i, c) not in done]
    if args.tts_limit:
        todo = todo[:max(0, args.tts_limit - len(done))]
    if not todo:
        log(f"speak: all {len(done)} recordings already transcribed")
        return True

    checkpoint = args.whistle_checkpoint or (TUNED_CHECKPOINT if os.path.exists(TUNED_CHECKPOINT)
                                             else DEFAULT_CHECKPOINT)
    log(f"speak: {len(todo)} recordings with {engine}, transcribed by Whistle ({checkpoint}, "
        f"beam {args.beam_size}, tool words at boost {args.tool_boost:g})")
    transcriber = Transcriber(checkpoint)
    transcriber.warmup(args.beam_size)
    keywords = whistle_settings(tools, args)
    rng, np_rng = random.Random(args.seed), np.random.default_rng(args.seed)
    workdir = tempfile.mkdtemp(prefix="dana-tts-")
    start, failures = time.time(), 0
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for n, (i, copy) in enumerate(todo, 1):
                text = examples[i]["query"]
                wav = os.path.join(workdir, "clip.wav")
                try:
                    synthesize(engine, text, wav, rng, args)
                    samples = read_wav(wav)
                except Exception as error:
                    failures += 1
                    if failures <= 3:
                        log(f"  TTS failed for {text!r}: {error}")
                    continue
                if np_rng.random() < args.augment:
                    samples = augment(samples, np_rng)
                heard = transcriber.transcribe(samples, language=args.language, keywords=keywords,
                                               boost=args.boost, beam_size=args.beam_size)["text"]
                f.write(json.dumps({"index": i, "copy": copy, "query": text, "heard": heard,
                                    "engine": engine}, ensure_ascii=False) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    rate = (time.time() - start) / n
                    log(f"  {n}/{len(todo)}  ({rate:.1f} s each, {rate * (len(todo) - n) / 60:.0f} min left)  "
                        f"{text!r} -> {heard!r}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if failures:
        log(f"  {failures} commands could not be spoken")
    return True


# ---------------------------------------------------------------- 3. build

def word_errors(ref, hyp):
    """Word-level edit distance between two normalised strings."""
    r, h = ref.split(), hyp.split()
    row = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, row[0] = row[0], i
        for j, hw in enumerate(h, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rw != hw))
    return row[-1], len(r)


def learn_confusions(pairs):
    """{phrase: [what Whistle wrote instead, ...]} and the words it tends to drop, from
    (said, heard) pairs. Phrases are one or two normalised words."""
    swaps, drops = collections.defaultdict(list), collections.Counter()
    for said, heard in pairs:
        a, b = norm_text(said).split(), norm_text(heard).split()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if op == "replace" and i2 - i1 <= 2 and j2 - j1 <= 3:
                swaps[" ".join(a[i1:i2])].append(" ".join(b[j1:j2]))
            elif op == "delete" and i2 - i1 == 1:
                drops[a[i1]] += 1
    return dict(swaps), drops


# Sound-alikes recognisers mix up, used when there is no TTS to learn real ones from.
GENERIC_SWAPS = {
    "to": ["two", "too"], "two": ["to"], "for": ["four"], "four": ["for"], "right": ["write"],
    "on": ["in", "and"], "off": ["of"], "the": ["a"], "a": ["the"], "in": ["and", "on"], "an": ["and"],
    "lights": ["light", "likes"], "light": ["lights", "right"], "set": ["said", "sit"], "turn": ["tune"],
    "fan": ["van", "fun"], "door": ["dog", "dorm"], "lock": ["look"], "close": ["clothes"],
    "music": ["muse"], "play": ["pay", "played"], "pause": ["paws", "pose"], "next": ["necks"],
    "volume": ["value"], "timer": ["time"], "minutes": ["minute"], "degrees": ["degree"],
    "brightness": ["brightest"], "kitchen": ["kitten"], "bedroom": ["bathroom"], "start": ["star"],
    "stop": ["stuff"], "call": ["cold"], "text": ["next"], "send": ["sent"], "open": ["often"],
}
FILLERS = {"the", "a", "to", "please", "my", "of", "and", "uh", "um", "now"}


def noisy_copy(query, swaps, drops, rng, strength=0.25):
    """`query` with recognition-like mistakes: learnt (or generic) swaps, dropped small
    words, lost punctuation and capitals. None if it came out unchanged."""
    words = query.split()
    out, changed, i = [], False, 0
    while i < len(words):
        word = words[i]
        core = norm_text(word)
        pair = " ".join(norm_text(w) for w in words[i:i + 2])
        if i + 1 < len(words) and swaps.get(pair) and rng.random() < strength:
            new = rng.choice(swaps[pair])  # a two-word mix-up Whistle made ("the blinds" -> "a blends")
            out.append(new[:1].upper() + new[1:] if word[:1].isupper() else new)
            changed = changed or new != pair
            i += 2
            continue
        i += 1
        options = swaps.get(core) or GENERIC_SWAPS.get(core)
        if options and rng.random() < strength:
            new = rng.choice(options)
            out.append(new[:1].upper() + new[1:] if word[:1].isupper() else new)
            changed = changed or norm_text(new) != core
        elif core in FILLERS | set(w for w, c in drops.items() if c > 1) and rng.random() < strength / 3:
            changed = True
        else:
            out.append(word)
    text = " ".join(out)
    if rng.random() < 0.3:
        text = re.sub(r"[,.!?;:]", "", text)
    if rng.random() < 0.2:
        text = text.lower()
    return text if changed and text.strip() else None


def copied_values(said, calls):
    """String argument values that appear word for word in what was said ("Ardour", "llama
    pictures"): Dana copies these from the text, so they must survive recognition."""
    said = f" {norm_text(said)} "
    out = []
    for call in calls:
        for value in (call.get("arguments") or {}).values():
            if isinstance(value, str) and norm_text(value) and f" {norm_text(value)} " in said:
                out.append(norm_text(value))
    return out


def recoverable(heard, values):
    """Whether every copied value is still in the (mis)heard text. When Whistle wrote "Mar key"
    for "Marqui", asking Dana for "Marqui" teaches it to invent names, not to copy them."""
    heard = f" {norm_text(heard)} "
    return all(f" {v} " in heard for v in values)


def build(examples, tools, heard_rows, args):
    tools_str = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    rng = random.Random(args.seed + 1)

    pairs, errors, words, kept_heard, dropped = [], 0, 0, collections.defaultdict(list), 0
    for row in heard_rows:
        said, heard = examples[row["index"]]["query"], row["heard"]
        e, n = word_errors(norm_text(said), norm_text(heard))
        errors, words = errors + e, words + n
        similar = difflib.SequenceMatcher(a=norm_text(said), b=norm_text(heard)).ratio()
        if not heard.strip() or similar < args.min_similarity:
            dropped += 1  # too garbled to be worth learning from
            continue
        pairs.append((said, heard))
        kept_heard[row["index"]].append(heard)
    stats = {}
    if heard_rows:
        stats["whistle_wer"] = errors / max(words, 1)
        log(f"build: Whistle word error rate on the spoken commands: {stats['whistle_wer']:.1%} "
            f"({dropped} of {len(heard_rows)} too garbled, left out)")
    swaps, drops = learn_confusions(pairs)
    if swaps:
        common = sorted(swaps.items(), key=lambda kv: -len(kv[1]))[:12]
        log("  most common mix-ups: " + "; ".join(f"{k} -> {collections.Counter(v).most_common(1)[0][0]}"
                                                 for k, v in common))

    rows, counts = [], collections.Counter()

    def add(query, example, source):
        rows.append({"query": query, "tools": tools_str,
                     "answers": json.dumps(example["answers"], separators=(",", ":"), ensure_ascii=False),
                     "split": example["split"], "source": source})
        counts[example["split"], source] += 1

    lost = collections.Counter()
    for i, example in enumerate(examples):
        add(example["query"], example, "clean")
        seen = {norm_text(example["query"])}
        values = copied_values(example["query"], example["answers"])
        for heard in kept_heard.get(i, []):
            if not recoverable(heard, values):
                lost["heard"] += 1
                continue
            if norm_text(heard) not in seen:
                seen.add(norm_text(heard))
                add(heard, example, "heard")
        if example["split"] != "train":
            continue  # val/test: only real recognition output, not made-up noise
        for _ in range(args.text_copies):
            noisy = noisy_copy(example["query"], swaps, drops, rng)
            if noisy and not recoverable(noisy, values):
                lost["noise"] += 1
                continue
            if noisy and norm_text(noisy) not in seen:
                seen.add(norm_text(noisy))
                add(noisy, example, "noise")
    if lost:
        log(f"  left out {lost['heard']} heard and {lost['noise']} noisy versions that garbled a value Dana "
            "must copy (it cannot know the real spelling)")
    stats["unrecoverable"] = dict(lost)
    rng.shuffle(rows)
    for split in ("train", "val", "test"):
        log(f"  {split}: " + ", ".join(f"{counts[split, s]} {s}" for s in ("clean", "heard", "noise")))
    stats["rows"] = {f"{k[0]}_{k[1]}": v for k, v in counts.items()}
    return rows, stats


# ---------------------------------------------------------------- 4/5. train and eval

def train(train_path, args):
    from dana.training.finetune import finetune_local

    result = finetune_local(argparse.Namespace(
        jsonl_path=train_path, checkpoint=args.base, epochs=args.epochs, batch_size=args.batch_size,
        checkpoint_dir=os.path.dirname(args.save) or "checkpoints", cache_dir=None,
        max_enc_len=None, max_dec_len=None))
    if not result or not result.get("checkpoint"):
        raise SystemExit("finetuning produced no checkpoint")
    shutil.copy2(result["checkpoint"], args.save)
    log(f"Saved {args.save}")
    return result


def evaluate(rows, args):
    from dana.dataset.dataset import get_tokenizer
    from dana.model.architecture import SimpleAttentionNetwork
    from dana.model.run import load_checkpoint
    from dana.training.finetune import _quick_tool_eval

    test = [r for r in rows if r["split"] == "test"]
    calling = [r for r in test if r["answers"].strip() != "[]"]
    none = [r for r in test if r["answers"].strip() == "[]" and r["source"] in ("clean", "heard")]
    groups = {"clean": [r for r in calling if r["source"] == "clean"],
              "heard": [r for r in calling if r["source"] == "heard"]}
    tokenizer, report = get_tokenizer(), {}
    for label, path in (("base", args.base), ("finetuned", args.save)):
        if not os.path.exists(path):
            continue
        params, config = load_checkpoint(path)
        model = SimpleAttentionNetwork(config)
        report[label] = {g: _quick_tool_eval(model, params, tokenizer, items) for g, items in groups.items() if items}
        if none:
            report[label]["none"] = no_call_rate(model, params, tokenizer, none)
        del params, model
    log("\nHeld-out commands       exact match   call F1")
    for label, by_group in report.items():
        for group, m in by_group.items():
            if group in groups:
                log(f"  {label:9s} {group:6s} ({len(groups[group]):3d})   {m.get('exact_match', 0):7.1%}   "
                    f"{m.get('call_f1', 0):7.1%}")
    if none:
        log(f"\nNo-tool requests ({len(none)})   no call made")
        for label, by_group in report.items():
            if "none" in by_group:
                log(f"  {label:9s}                {by_group['none']['no_call']:7.1%}")
    return report


def no_call_rate(model, params, tokenizer, rows):
    """Share of `rows` (requests no tool can carry out) for which Dana makes no call."""
    from dana.model.run import generate_batch

    quiet = 0
    for i in range(0, len(rows), 32):
        chunk = rows[i:i + 32]
        for text in generate_batch(model, params, tokenizer, [r["query"] for r in chunk],
                                   [r["tools"] for r in chunk], constrained=True):
            try:
                calls = json.loads(text)
            except json.JSONDecodeError:
                calls = None
            quiet += calls == []
    return {"no_call": quiet / len(rows), "n": len(rows)}


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tools", help="Tools JSON file, as `dana voice --tools` takes it")
    p.add_argument("--name", help="Short name for outputs (default: the tools file's name)")
    p.add_argument("--out", help="Working folder (default: data/voice/<name>)")
    p.add_argument("--save", help="Finetuned checkpoint (default: checkpoints/dana_<name>.pkl)")
    p.add_argument("--stages", default=",".join(STAGES), help=f"Stages to run, comma separated: {','.join(STAGES)}")
    p.add_argument("--redo", action="append", default=[], choices=STAGES, help="Run this stage again even if done")
    p.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("1. generate")
    g.add_argument("--data", help="Your own JSONL of commands and calls, instead of asking Gemini")
    g.add_argument("--add-data", help="JSONL of more commands to append to an existing examples.jsonl; earlier "
                                      "commands keep their splits and recordings, only the new ones are spoken")
    g.add_argument("--per-tool", type=int, default=150, help="Commands to ask for per tool (default: 150)")
    g.add_argument("--multi", type=float, default=0.1,
                   help="Extra commands asking for 2-3 things at once, as a fraction of the total (default: 0.1)")
    g.add_argument("--gemini-model", default=None, help="Gemini model (default: the one dana generate-data uses)")
    g.add_argument("--workers", type=int, default=4, help="Parallel Gemini calls")
    g = p.add_argument_group("2. speak")
    g.add_argument("--tts", default="auto", choices=["auto", "kokoro", "piper", "espeak-ng", "espeak", "pico2wave", "none"])
    g.add_argument("--kokoro-model", default=os.path.join("checkpoints", "kokoro", "kokoro-v1.0.onnx"))
    g.add_argument("--kokoro-voices", default=os.path.join("checkpoints", "kokoro", "voices-v1.0.bin"))
    g.add_argument("--piper-model", nargs="+", default=None, help="Piper voice .onnx file(s); one is picked per clip")
    g.add_argument("--piper-cmd", default="piper")
    g.add_argument("--tts-copies", type=int, default=1, help="Recordings per command, each a different voice")
    g.add_argument("--tts-limit", type=int, default=None, help="Speak at most this many recordings in all")
    g.add_argument("--augment", type=float, default=0.7, help="Share of clips given random noise/speed/volume")
    g.add_argument("--language", default="en")
    g.add_argument("--whistle-checkpoint", default=None,
                   help="Whistle weights (default: as dana voice: whistle_dana if present)")
    g.add_argument("--beam-size", type=int, default=4, help="As dana voice --beam-size")
    g.add_argument("--tool-boost", type=float, default=2.0, help="As dana voice --tool-boost")
    g.add_argument("--max-tool-keywords", type=int, default=300)
    g.add_argument("--boost", type=float, default=3.0)
    g = p.add_argument_group("3. build")
    g.add_argument("--text-copies", type=int, default=1, help="Made-up noisy copies per training command")
    g.add_argument("--min-similarity", type=float, default=0.5,
                   help="Leave out transcripts less similar than this to what was said (0-1)")
    g = p.add_argument_group("4. train")
    g.add_argument("--base", default=os.path.join("checkpoints", "dana.pkl"), help="Dana checkpoint to start from")
    g.add_argument("--epochs", type=int, default=2)
    g.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    if not os.path.isdir("checkpoints"):
        log("Note: no checkpoints/ folder here; run this from the repository folder, as `dana` commands are")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")
    tools_path = args.tools
    tools = load_tools(tools_path)
    name = args.name or os.path.splitext(os.path.basename(tools_path))[0]
    out = args.out or os.path.join("data", "voice", name)
    args.save = args.save or os.path.join("checkpoints", f"dana_{name}.pkl")
    os.makedirs(out, exist_ok=True)
    paths = {s: os.path.join(out, f) for s, f in (("generate", "examples.jsonl"), ("speak", "heard.jsonl"),
                                                 ("build", "train.jsonl"))}
    for stage in args.redo:
        if stage in paths and os.path.exists(paths[stage]):
            os.remove(paths[stage])
    report_path = os.path.join(out, "report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    report.update({"tools": tools_path, "name": name})
    log(f"{len(tools)} tools from {tools_path}; working in {out}")

    examples = read_jsonl(paths["generate"]) if os.path.exists(paths["generate"]) else None
    if "generate" in stages and examples is None:
        examples = read_data(args.data, tools) if args.data else generate(tools, args)
        examples = assign_splits(examples, args.seed)
        write_jsonl(paths["generate"], examples)
        per = collections.Counter(primary(e["answers"]) for e in examples)
        log(f"generate: {len(examples)} commands -> {paths['generate']}")
        thin = {k: v for k, v in per.items() if v < 120}
        if thin:
            log("  fewer than 120 commands for: " + ", ".join(f"{k} ({v})" for k, v in sorted(thin.items())))
    if args.add_data:
        if examples is None:
            raise SystemExit(f"--add-data appends to {paths['generate']}, which does not exist yet")
        have = {norm_text(e["query"]) for e in examples}
        new = [e for e in read_data(args.add_data, tools) if norm_text(e["query"]) not in have]
        new = assign_splits(new, args.seed + len(examples))
        examples += new
        write_jsonl(paths["generate"], examples)
        if os.path.exists(paths["build"]):
            os.remove(paths["build"])  # stale now; rebuilt below
        if new and "train" not in args.redo:
            args.redo.append("train")  # the old checkpoint did not see the new commands
        log(f"add-data: {len(new)} new commands appended -> {len(examples)} in {paths['generate']}")
    if examples is None and any(s in stages for s in ("speak", "build")):
        raise SystemExit(f"{paths['generate']} is missing: run the generate stage first")

    if "speak" in stages:
        speak(examples, tools, args, paths["speak"])

    rows = read_jsonl(paths["build"]) if os.path.exists(paths["build"]) else None
    if "build" in stages and rows is None:
        heard = read_jsonl(paths["speak"]) if os.path.exists(paths["speak"]) else []
        rows, stats = build(examples, tools, heard, args)
        write_jsonl(paths["build"], rows)
        report["data"] = stats
        log(f"build: {len(rows)} rows -> {paths['build']}")

    if "train" in stages:
        if rows is None:
            raise SystemExit(f"{paths['build']} is missing: run the build stage first")
        if os.path.exists(args.save) and "train" not in args.redo:
            log(f"train: {args.save} exists, skipping (--redo train to train again)")
        else:
            result = train(paths["build"], args)
            report["train"] = {"checkpoint": args.save, "test_all": {k: result[k] for k in ("base", "finetuned")}}

    if "eval" in stages and rows is not None:
        report["eval"] = evaluate(rows, args)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    if os.path.exists(args.save):
        log(f"\nDone. Try it:\n  dana voice --tools {args.tools} --dana-checkpoint {args.save}")


if __name__ == "__main__":
    main()
