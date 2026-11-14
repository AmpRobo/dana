> **独立版说明**：这是 Cactus Needle 1 的独立副本，本地重命名为 Dana（上游 `cactus-compute/cactus@a5af1f8` + 官方权重），做了本地化改动。中文使用说明和改动列表见 [GUIDE_zh.md](GUIDE_zh.md)。

# Dana

<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 100%;">

A 26m parameter "[Simple Attention Network](https://arxiv.org/abs/2607.18363)" for function calling that you can even finetune locally on your Mac/PC.
In production, Dana runs on [Cactus](https://github.com/cactus-compute/cactus) at 6000 toks/sec prefill and 1200 decode speed. 
Weights and tokenizer are bundled locally (`checkpoints/dana.pkl`, `dana/tokenizer/`); nothing is fetched from Hugging Face.

```
d=512, 8H/4KV, BPE=8192
                                  ┌──────────────┐
                                  │  Tool Call   │
                                  └──────┬───────┘
                                        ┌┴──────────┐
                                        │  Softmax  │
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ Linear (T)│  ← tied
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ ZCRMSNorm │
                                        └─────┬─────┘
                                     ┌────────┴────────┐
                                     │ Decoder x 8     │
                                     │┌───────────────┐│
                                     ││ ZCRMSNorm     ││
                                     ││ Masked Self   ││
                                     ││ Attn + RoPE   ││
                                     ││ Gated Residual││
                                     │├───────────────┤│
  ┌──────────────┐                   ││ ZCRMSNorm     ││
  │ Encoder x 12 │──────────────────────▶Cross Attn   ││
  │              │                   ││ Gated Residual││
  │ ┌──────────┐ │                   │└───────────────┘│
  │ │ZCRMSNorm │ │                   └────────┬────────┘
  │ │Self Attn │ │                      ┌─────┴─────┐
  │ │ GQA+RoPE │ │                      │ Embedding │  ← shared
  │ │Gated Res │ │                      └─────┬─────┘
  │ │          │ │                    ┌───────┴───────-┐
  │ │ (no FFN) │ │                    │[EOS]<tool_call>│
  │ └──────────┘ │                    │ + answer       │
  │              │                    └───────────────-┘
  └──────┬───────┘
         │
    ┌────┴──────┐
    │ Embedding │
    └────┬──────┘
         │
    ┌────┴──────┐
    │   Text    │
    │  query    │
    └───────────┘
```

- Pretrained on 16 TPU v6e for 200B tokens (27hrs). 
- Post-trained on 2B tokens of single-shot function call dataset (45mins). 

Dana is an experimental run for Simple Attention Networks, geared at redefining tiny AI for consumer devices (phones, watches, glasses...).
So while it beats FunctionGemma-270m, Qwen-0.6B, Graninte-350m, LFM2.5-350m on single-shot function call for personal AI,
Those model are have more scope/capacity and excel in conversational settings. Also, small models can be finicky. 
Please use the UI in the next section to test on your own tools, and finetune accordingly, at the click of a button. 

## 本地测试运行

所有命令都在项目目录里运行，先激活虚拟环境：

```bash
cd /home/xiaoyu/Documents/code/dana && source .venv/bin/activate
```

**1. 单条推理（最快）**

```bash
dana run --checkpoint checkpoints/dana.pkl --query "What's the weather in Paris?" --tools '[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]'
```

也可以从 `examples/tools/` 里现成的工具文件读工具（33 个类别，比如 finance、smart_home、navigation）：

```bash
dana run --checkpoint checkpoints/dana.pkl --query "send 20 dollars to Alice for dinner" --tools "$(cat examples/tools/finance.json)"
```

预期输出类似 `[{"name":"send_payment","arguments":{"contact_id":"contact_alice_123","amount":20,...}}]`。

**2. 浏览器 playground（交互式）**

```bash
dana playground
```

打开 http://127.0.0.1:7860，可以试自己的工具，也可以一键微调。默认用 `checkpoints/dana.pkl`，用 `--port` 换端口。

**3. 在 Python 里调用**

```python
from dana import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer
params, config = load_checkpoint("checkpoints/dana.pkl")
model = SimpleAttentionNetwork(config)
print(generate(model, params, get_tokenizer(), query="...", tools='[...]', stream=False))
```

**4. 用自己的数据微调**

```bash
dana finetune data.jsonl
```

JSONL 每行需要 `query`、`tools`、`answers` 三个字段（`tools` 和 `answers` 是 JSON 字符串），格式见下面的 [Data format](#data-format)。最好的 checkpoint 会存成 `checkpoints/dana_finetuned_<id>_best.pkl`。

**5. 语音转文字（Whistle）**

```bash
dana transcribe clip.wav                    # 自动识别语言
dana transcribe clip.wav --language de      # 指定语言：en、de、fr、es、it、nl、pl
dana transcribe a.wav b.wav --json          # 多个文件，每个输出一行 JSON
dana transcribe clip.wav --keywords "Dana,Xiaoyu"   # 偏向这些词（人名、产品名等）
```

`--keywords` 不用训练：解码时给这些词加分，模型听着像的时候就写成你给的拼写（比如 `--keywords Center` 会把 Front_Center.wav 的 “Centre” 改成 “Center”）。词要按想看到的大小写和拼写写，多个词用逗号隔开。默认强度 `--boost 3`；还是识别不出来就试 `--boost 5`，到 8 左右开始会把别的词也替换掉。

WAV 文件任意采样率、单声道或立体声都行，超过 30 秒会按 30 秒分段。使用 `checkpoints/whistle.safetensors` 里的官方权重（或用 `--checkpoint` 指定），不会自动下载。可以先用系统自带的测试语音试一下：`dana transcribe /usr/share/sounds/alsa/Front_Center.wav`。Whistle 是独立的语音模型，和 Dana 不共享权重、分词器或模型代码，详见 [GUIDE_zh.md](GUIDE_zh.md)。

**6. 麦克风实时识别**

```bash
dana listen                     # 默认麦克风，Ctrl+C 结束
dana listen --language en       # 固定语言，比每句自动识别更稳
dana listen --keywords "Dana,Xiaoyu"   # 关键词偏置，用法同上
```

`dana listen` 默认就带 `--keywords Dana`，听到像 “Dana” 的音会写成 “Dana”。自己指定 `--keywords` 会替换掉默认值，要保留就把 Dana 也写上；`--keywords ""` 关掉。

启动后前 1 秒测房间底噪（这时别说话），之后每说完一句、停顿 0.8 秒，就会输出这一句的识别结果（CPU 上一句话大约 0.5 秒）。用系统的 `arecord` 录音，不需要装额外的包。识别不到声音时，先用 `arecord -l` 看麦克风编号，再用比如 `--device plughw:0,6` 指定；环境吵或者说话被截断时，可以调 `--threshold`（比如 0.02）和 `--silence`（比如 1.2）。

**7. 教 Whistle 认一个新词（以 Dana 为例）**

```bash
dana record "Dana." --count 5 --out data/whistle/dana     # 每次出现提示就说一遍，说完停一秒
python scripts/whistle_keyword_data.py data/whistle/dana/clips.jsonl --out data/whistle/dana_train \
    --context /usr/share/sounds/alsa/[FRS]*.wav /usr/share/sounds/speech-dispatcher/*.wav
dana finetune-whistle data/whistle/dana_train/train.jsonl --validation data/whistle/dana_train/val.jsonl \
    --out checkpoints/whistle_dana.safetensors
dana listen --language en                                  # 有 whistle_dana 就自动用它
```

第二步把你的录音加上随机音量、噪声、语速和前后留白，生成三类训练样本：
- 单独说这个词；
- 这个词接在别的话前后；
- 只有别的话，用原模型自己的识别结果当标注，免得它把别的也学歪。

最后一条录音只用来验证。第三步是 LoRA 微调，CPU 上大约 4 分钟，结束后打印验证集的识别结果。`dana listen` 会优先加载 `checkpoints/whistle_dana.safetensors`，要用原版权重就加 `--checkpoint checkpoints/whistle.safetensors`。效果不稳时，多录几遍（换个距离、语气，或者录 “Hi Dana.” 这样的句子），然后重跑第二、三步。如果它把别的词也识别成 Dana，就用 `dana record` 录一些你说的其他话，加到 `--context` 里一起训练。

**8. 用语音调用 tools（Whistle + Dana）**

```bash
dana voice --tools examples/tools/smart_home.json
```

说 “Dana, turn on the kitchen lights.”：Whistle 把这句话转成文字，“Dana” 是唤醒词，后面那部分交给 Dana 模型，再配上 tools 生成调用。也可以先说 “Dana.”，停一下，在 10 秒内（`--wait`）说命令。

每条命令在 stdout 输出一行 JSON：

```json
{"command": "Turn on the kitchen lights.", "calls": [{"name": "control_lights", "arguments": {"room": "kitchen", "action": "on"}}]}
```

听到的文字和耗时打印在 stderr，所以可以用管道把调用交给自己的程序执行，比如 `dana voice --tools tools.json | python my_executor.py`。`dana voice` 本身只生成调用，不会执行任何操作。

- **唤醒词**：没以唤醒词开头的话只显示、不当命令。因为 Dana 对任何输入都会编出一个调用，比如 “Front left.” 会变成 `control_blinds`，所以默认需要唤醒词。`--wake ""` 让每句话都当命令。
- **tools**：`--tools` 可以是 `examples/tools/` 里的文件，也可以直接写 JSON。`examples/tools/desktop.json` 有 54 个桌面操作（开关应用和窗口、音量、亮度、锁屏和关机、Wi-Fi、文件夹、网页、打字和快捷键等），可以用语音控制电脑，执行部分要自己写。
- **tools 的长度**：Dana 最多读 4096 token（大约 110 个工具），启动时会打印 tools 占多少 token，超出时提醒。
- **语言**：默认 `--language en`，因为 Dana 只认英文。
- **速度**：Dana 生成调用在 CPU 上每条 0.4–0.6 秒（`smart_home.json`，约 750 token），`desktop.json`（约 1850 token）0.8–1.3 秒。大部分时间花在读 tools 上，tools 越长越慢，接近 4096 token 时光读 tools 就要 2 秒多。

**暂时跑不了的：** `dana eval`、`dana train`、`dana tokenize` 需要 `data/tool_calls_unified/` 下的工具调用数据集，本地还没有。要先用 `dana generate-data` 生成，这一步需要 `GEMINI_API_KEY`。

## Quickstart

### macOS / Linux

```bash
git clone https://github.com/cactus-compute/needle.git
cd dana && source ./setup
dana playground
```

### Windows

```bat
git clone https://github.com/cactus-compute/needle.git
cd dana
.\setup.bat
```

> **Note:** `setup.bat` creates a `.venv`, installs all dependencies (including JAX in CPU mode), and prompts for an optional W&B API key.

After setup completes, activate the virtual environment in every new terminal session before running dana commands:

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Tip:** If PowerShell blocks the script, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once and retry.

Then launch the playground:

```powershell
dana playground
```

Opens a web UI at http://127.0.0.1:7860 where you can test and finetune on your own tools. Weights are loaded from `checkpoints/dana.pkl`.

## Usage (Python)

```python
from dana import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer

params, config = load_checkpoint("checkpoints/dana.pkl")
model = SimpleAttentionNetwork(config)
tokenizer = get_tokenizer()

result = generate(
    model, params, tokenizer,
    query="What's the weather in San Francisco?",
    tools='[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]',
    stream=False,
)
print(result)
# [{"name":"get_weather","arguments":{"location":"San Francisco"}}]
```

## Finetuning

```bash
# Playground (generates data via Gemini, trains, evaluates, bundles result)
dana playground

# CLI (uses checkpoints/dana.pkl unless --checkpoint is given)
dana finetune data.jsonl
```

### Data format

Each line in the JSONL file has three fields: `query`, `tools`, and `answers`.

**Tool schema:**
```json
{
  "name": "get_weather",
  "description": "Get current weather for a city.",
  "parameters": {
    "location": { "type": "string", "description": "City name.", "required": true }
  }
}
```

**Answer schema:**
```json
{ "name": "get_weather", "arguments": { "location": "Paris" } }
```

**Full JSONL example** (each line is one training example, `tools` and `answers` are JSON-encoded strings):
```jsonl
{"query": "What's the weather in Paris?", "tools": "[{\"name\":\"get_weather\",\"description\":\"Get current weather for a city.\",\"parameters\":{\"location\":{\"type\":\"string\",\"description\":\"City name.\",\"required\":true}}}]", "answers": "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}]"}
{"query": "Turn off the lights", "tools": "[{\"name\":\"get_weather\",\"description\":\"Get current weather for a city.\",\"parameters\":{\"location\":{\"type\":\"string\",\"description\":\"City name.\",\"required\":true}}},{\"name\":\"toggle_lights\",\"description\":\"Toggle smart lights on or off.\",\"parameters\":{\"state\":{\"type\":\"string\",\"description\":\"on or off.\",\"required\":true}}}]", "answers": "[{\"name\":\"toggle_lights\",\"arguments\":{\"state\":\"off\"}}]"}
```

Provide at least **120 examples per tool** (100 train / 10 val / 10 test). Fewer examples will overfit — you'll see perfect training metrics but the model won't generalize. Vary query phrasing and include examples with multiple tools available.

### Using a finetuned model

Finetuning saves the best checkpoint as `checkpoints/dana_finetuned_<id>_best.pkl`:

```bash
dana run --checkpoint checkpoints/dana_finetuned_*_best.pkl \
  --query "What's the weather?" --tools '[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]'
```

```python
params, config = load_checkpoint("checkpoints/dana_finetuned_<id>_best.pkl")
model = SimpleAttentionNetwork(config)
result = generate(model, params, get_tokenizer(), query="...", tools='[...]', stream=False)
```

## CLI

```
dana playground                  Test and finetune via web UI
dana finetune <data.jsonl>       Finetune on your own data
dana run --query "..." --tools   Single inference
dana train                       Full training run
dana pretrain --data <parquet>   Pretrain on local SYNTH-format parquet
dana eval --checkpoint <path>    Evaluate a checkpoint
dana transcribe <audio.wav>      Speech to text with Whistle
dana listen                      Live speech to text from the microphone
dana voice --tools <tools.json>  Spoken commands to tool calls (Whistle + Dana)
dana record "<text>"             Record labelled clips for finetuning Whistle
dana finetune-whistle <train>    LoRA-finetune Whistle on JSONL clips
dana tokenize                    Tokenize dataset
dana generate-data               Synthesize training data via Gemini
dana tpu <action>                TPU management (see docs/tpu.md)
```

```
@misc{ndubuaku2026needle,
  title={Needle},
  author={Henry Ndubuaku, Jakub Mroz,  Karen Mosoyan, Roman Shemet, Parkirat Sandhu, Satyajit Kumar, Noah Cylich, Justin H. Lee},
  year={2026},
  url={https://github.com/cactus-compute/needle}
}
```
