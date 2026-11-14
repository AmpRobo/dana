# Dana 1 独立版使用说明

这是 Cactus **Needle 1** 的完整独立副本，本地整体重命名为 **Dana**：26M 参数、纯注意力（没有 FFN）的 encoder-decoder 函数调用模型，附带完整的推理、微调、预训练、后训练代码，以及官方发布的权重。

| 来源 | 版本 |
|---|---|
| 代码 | `github.com/cactus-compute/cactus` 的提交 `a5af1f8`（2026-08-09）。这是 Needle 1 最后一个完整版本，下一个提交就开始换成 Needle 2 的说明了 |
| 权重和分词器 | Hugging Face `Cactus-Compute/needle` 的 revision `5f89b43`，sha256 已校验 |
| Whistle 代码 | `github.com/cactus-compute/cactus` 的提交 `bafcf73`，移植到 `dana/whistle/`（Apache-2.0） |
| Whistle 权重 | Hugging Face `Cactus-Compute/whistle` 的 revision `7411186`，一次性手动下载，sha256 `5fd58c24…d107` 已校验 |
| 许可证 | MIT（见 `LICENSE`）；`dana/whistle/` 是 Apache-2.0 |

英文原版说明在 [README.md](README.md)，设计文档在 [docs/simple_attention_networks.md](docs/simple_attention_networks.md)，TPU 指南在 [docs/tpu.md](docs/tpu.md)。

## 目录

```
dana/
├── checkpoints/dana.pkl      官方 Dana 1 权重（52.6 MB）
├── checkpoints/whistle.safetensors  官方 Whistle 权重（220.6 MB）
├── checkpoints/whistle_dana.safetensors  微调后能认出 “Dana” 的 Whistle（dana listen 默认用它）
├── data/whistle/dana/        录的 5 遍 “Dana” 和 clips.jsonl；dana_train/ 是生成的训练数据
├── dana/tokenizer/           SentencePiece 分词器，词表 8192
├── dana/model/               架构、推理、约束解码、导出、量化
├── dana/training/            pretrain / train / finetune / eval / optim(Muon)
├── dana/dataset/             数据生成(Gemini)、分词、打包
├── dana/ui/                  浏览器 playground
├── dana/utils/               多机分布式、TPU 管理
├── dana/whistle/             Whistle 语音转文字（独立模块，自带分词器）
├── setup / setup.bat           一键环境安装
├── launch_train.sh             多主机 TPU 启动脚本
└── docs/                       设计文档、TPU 指南、本地改动 diff
```

## 1. 安装

```bash
cd /home/xiaoyu/Documents/code/dana
```

```bash
source ./setup
```

`setup` 会创建 `.venv`，自动识别 GPU/TPU/CPU 并安装对应的 jax，最后会问 W&B key（直接回车可以跳过）。想手动指定可以加 `DANA_ACCELERATOR=cpu|gpu|tpu`。

如果 `.venv` 已经建好，以后每次打开新终端只需要：

```bash
source .venv/bin/activate
```

## 2. 推理

```bash
dana run --checkpoint checkpoints/dana.pkl --query "What's the weather in Paris?" --tools '[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]'
```

浏览器 playground（地址 http://127.0.0.1:7860）：

```bash
dana playground
```

Python 调用：

```python
from dana import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer
params, config = load_checkpoint("checkpoints/dana.pkl")
model = SimpleAttentionNetwork(config)
print(generate(model, params, get_tokenizer(),
               query="What's the weather in Paris?",
               tools='[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]',
               stream=False))
```

工具格式是 Dana 1 自己的写法：`parameters` 下面直接写 `{参数名: {type, description, required}}`，不是 JSON Schema。

**输入长度**：Dana 训练时 encoder 最长 1024 token，推理时上限提高到 4096（`DEFAULT_RUN_ENC_LEN`，在 `dana/dataset/tokenizer.py`），再长的部分仍然截掉。超过 1024 时，RoPE 改用 YaRN 缩放（`dana/model/architecture.py` 的 `precompute_rope_freqs`，按 `config.max_seq_len` 判断训练长度）：在训练长度内转不满一圈的低频按比例放慢，留在训练见过的角度里；转得快的高频不变；再把注意力调尖一点，抵消输入变长带来的稀释。1024 以内的输入和以前完全一样。`train`、`tokenize`、`eval` 仍然按 1024。

用 `examples/tools/desktop.json`（54 个工具，1846 token）和 51 条桌面命令测试（`dana.pkl`，贪心解码）：

| 做法 | 选对工具 | 工具和参数都对 |
|---|---|---|
| 截到 1024 token（以前的做法，后一半工具看不到） | 23/51 | 23/51 |
| 读完 1846 token，RoPE 不缩放 | 5/51 | 4/51 |
| 位置插值（所有频率一起放慢） | 40/51 | 34/51 |
| **YaRN（现在的做法）** | **46/51** | **45/51** |
| 参照：工具拆成两半（各不到 1024），每条命令只看有答案的那一半 | 47/51 | 47/51 |

再加 32 个其他类别的工具凑到 3818 token：desktop 放前面 42/51（参数也对 39/51），放后面 38/51（36/51）。所以 2000 token 左右基本不掉，接近 4096 时会掉一些，工具越靠后越容易选错。要在 4096 上更准，需要按 `scripts/longctx_finetune.py` 在长输入上微调；那要先跑完 `scripts/longctx_label.py` 标数据，CPU 上训练很慢。

**解码速度**：`generate()` 和 `generate_batch()` 用 key/value cache 解码（`dana/model/kv_decode.py`）：encoder 输出的 cross-attention key/value 每条输入只算一次，之后每一步只算新生成的那个 token。以前每生成一个 token，都要把 decoder 的 512 个位置和 encoder 输出的 key/value 全部重算一遍，tools 越长越慢。数学和原来的 decoder 一样：float32 下 logits 只差 3e-5（原 decoder 换个 buffer 长度跑也差这么多），bf16 下每个位置的 argmax 都一致，上面 desktop 的几组测试结果也和以前完全相同。encoder 也加了 jit；输入补齐到 256 的倍数，同一档长度只编译一次。CPU 上单条命令：

| tools | 以前 | 现在 |
|---|---|---|
| `smart_home.json`（约 750 token） | 2.5–5 秒 | 0.4–0.6 秒 |
| `desktop.json`（1846 token） | 5–6.5 秒 | 0.8–1.3 秒 |

现在大部分时间是 encoder 读 tools：1846 token 约 0.6 秒，4096 token 约 2.2 秒（注意力随长度平方增长）。有 FFN 的模型（Dana 1 没有）仍然每步重算整个 decoder。

## 3. 在自己的数据上微调

数据是 JSONL，每行有 `query`、`tools`、`answers` 三个字段（`tools` 和 `answers` 是 JSON 字符串），格式见 README.md。官方建议每个工具至少准备 120 条（100 条训练 / 10 条验证 / 10 条测试）。

```bash
dana finetune data.jsonl
```

最好的那个 checkpoint 会存成 `checkpoints/dana_finetuned_<id>_best.pkl`，用 `dana run --checkpoint` 加载即可。

## 4. 从零完整训练

官方流程是：预训练 → 生成工具调用数据 → 分词打包 → 后训练 → 评测。

### 4.1 预训练（本地 parquet，SYNTH 格式）

只读本地数据，`--data` 必填（路径、目录或 glob）。数据格式是官方用的 SYNTH 格式，需要自己准备好 parquet：

```bash
dana pretrain --data "$HOME/data/synth/*.parquet" --wandb
```

- 总行数直接从 parquet 文件里数，学习率调度按实际数据量排。可以用 `--max-steps` 截短。
- 只读 `query`、`query_seed_text`、`synthetic_answer` 三列。SYNTH 每个文件是一个约 15.5 万行的 row group，第一个 batch 要等一整个文件读完。
- checkpoint 存在 `checkpoints/dana_base.pkl`，每 `--save-every` 步覆盖一次。续训用 `--checkpoint checkpoints/dana_base.pkl`。
- 学习率会按设备数自动放大：Adam 乘设备数，Muon 乘 √设备数。
- 默认配置（d512，12 层 encoder、8 层 decoder，序列 1024/512，bf16）就是官方的 26M。在 CPU 上建议加 `--dtype float32`，并把模型和序列都调小。

官方规模是 200B tokens，在 16 块 TPU v6e 上跑了 27 小时。没有 NVIDIA GPU 或 TPU 的话，只适合做缩小版实验。

### 4.2 生成后训练数据（需要 Gemini API key，会产生费用）

```bash
export GEMINI_API_KEY=...
```

```bash
dana generate-data --num-samples 5000
```

数据写到 `dana/data/tool_calls_unified/{train,validation}`。按 query 的哈希，大约每 50 条分 1 条到验证集；多次运行会累加并去重。加 `--dry-run` 只生成不保存。

官方当时用的数据集 `Cactus-Compute/tool-calls` 是私有的，这里只能自己重新生成。

### 4.3 分词打包

```bash
dana tokenize
```

- 读 4.2 生成的数据，打包结果写到 `dana/.data_cache/`。
- 默认**保留现有分词器**，这样后训练的结果和官方权重、和你自己预训练出来的 checkpoint 都还能对得上。只有从零重新预训练时才考虑加 `--train-tokenizer`，加了之后官方权重就不能再用了。

### 4.4 后训练

```bash
dana train --init-from checkpoints/dana_base.pkl --wandb
```

- 损失按 token 类型加权（工具名 / 参数值 / 参数 key 分别对应 `--w-name`、`--w-value`、`--w-key`），另外叠加 query↔tool 的对比学习损失（`--contrastive-weight`）。
- 最好的 checkpoint 存成 `checkpoints/<name>_12_512_best.pkl`。

### 4.5 评测

```bash
dana eval --checkpoint checkpoints/baseline_12_512_best.pkl
```

### 多机 TPU

见 [docs/tpu.md](docs/tpu.md)。常用的是 `dana tpu create/setup/sync/pretrain/train`。上游版本里 `tpu setup/sync` 打包目录的 bug 已经修好了。

### 4.6 语音转文字（Whistle）

Whistle 是 Cactus 的语音识别模型：7 种语言（en、de、fr、es、it、nl、pl），log-mel 特征 → 卷积下采样 → 8 层编码器 → 8 层解码器（交叉注意力），约 5500 万参数。

```bash
dana transcribe clip.wav
dana transcribe clip.wav --language fr --checkpoint checkpoints/whistle.safetensors
```

```python
from dana.whistle import Transcriber
t = Transcriber("checkpoints/whistle.safetensors")
print(t.transcribe("clip.wav"))   # {"text": ..., "language": ...}
```

- **独立模块**：`dana/whistle/` 是从 cactus `bafcf73` 移植的，不 import Dana 的模型代码，也不依赖 Needle 3 的代码。它用到的约 260 行基础模块（注意力、Engram、HadamardMLP、多车道残差等）都放在 `dana/whistle/layers.py` 里。
- **只做浮点推理**：去掉了量化感知训练和导出到原生运行时的分支，参数名和结构保持不变，所以官方 checkpoint 可以直接加载。
- **分词器**：官方权重是用 Needle 3 的分词器训练的，和 Dana 的分词器完全不同，所以这个分词器文件作为 Whistle 自己的词表放在 `dana/whistle/tokenizer/whistle.model`。
- **权重**：`checkpoints/whistle.safetensors` 是官方权重（训练时为 bfloat16，这里默认按 float32 推理）。代码不会自动下载，缺文件直接报错。
- **实测**：ALSA 自带的测试语音（`/usr/share/sounds/alsa/Front_Center.wav` 等）能正确识别成 “Front, Centre.”、“Front left.”；30 秒左右的长句基本正确，60 秒的音频会分两段识别。没有人声的音频（比如纯音乐）会被识别出没有意义的文字。
- **解码**：JAX 贪心解码，没有 KV cache；超过 30 秒的音频按 30 秒切段，切点可能落在词中间。没有移植词级时间戳和流式识别（这些只在 Cactus 原生运行时里有）。
- **关键词偏置**：`--keywords "Dana,Xiaoyu"`（`transcribe` 和 `listen` 都有；`listen` 默认是 `Dana`，`--keywords ""` 关掉），Python 里是 `t.transcribe(path, keywords=["Dana"], boost=3.0)`。Cactus 的关键词偏置在它的原生运行时里，没有 Python 代码可搬，这里是自己写的（`KeywordBias`，在 `transcribe.py`）：
  - 把每个关键词按 Whistle 的分词器切成 token 序列。这个分词器不会自动在开头加 “▁”，所以每个词准备两条：开头的写法（`Center`）和句中的写法（`▁Center`）。
  - 贪心解码的每一步，如果已输出的文字以某个关键词的前 m 个 token 结尾（m = 0 也算），就给它的下一个 token 的 logit 加 `--boost`（默认 3）。
  - 关键词只在模型自己要开始一个新词时才能起头，不能顶掉标点或结束符。否则会出现吞掉逗号、或者结尾不停重复 “Center Center …” 的情况。
  - 如果是加分让模型开始写一个关键词，但模型后面没写完，就退回到这个词的第一个 token，那个位置不加分重新解码，所以不会留下半个词。
- **关键词偏置实测**（只在这几个文件上测过）：
  - `Center`：Front_Center.wav 从 “Centre” 变成 “Center”，`--boost 1.5` 就够。
  - `speech-dispatcher`：speech-dispatcher 的 dummy-message.wav 里 6 处 “speech this patcher” 要 `--boost 5` 才全部改对。
  - 诱饵词：把 Xiaoyu、Dana、Frankfurt、Sidecar 等音频里没说的词当关键词，boost 到 5 都没有插进任何文件。到 8 开始替换别的词，比如 “1000” 变成 “Xiaoyu000”。
  - 开销：默认强度下解码步数多了约 7%。
- **麦克风实时识别**：`dana listen` 用 `arecord` 以 16 kHz 单声道录音，每 100 ms 算一次音量。前 1 秒测底噪，阈值取底噪的 3 倍；声音超过阈值算开始说话，安静满 `--silence` 秒（默认 0.8）算一句结束，然后识别这一句，录音在后台继续。一句最长 30 秒，短于 0.3 秒的响声（咳嗽、敲击）会被忽略。也可以用 `--stdin` 从管道读原始音频：`arecord -f S16_LE -r 16000 -c 1 -t raw | dana listen --stdin`。这不是逐字流式识别：要等一句说完才出结果。
- **微调认新词**：`dana record` 录带标注的短句，`scripts/whistle_keyword_data.py` 生成训练数据，`dana finetune-whistle` 做 LoRA 微调（README 第 7 步）。
  - **来源**：微调代码改编自 cactus 的 `whistle_finetune.py`。原版的量化感知训练、扩词表、新增语言和导出 `.cact` 都去掉了，只保留 float32 训练。
  - **训练方式**：在编码器和解码器各 8 层的 q/k/v/gate/out 投影上加 rank 16 的 LoRA（`--decoder-only` 只训练解码器），训练完合并成一个完整的 checkpoint。
  - **padding**：所有样本补齐到同一长度，只编译一次。补到 4 秒和补到 30 秒，logits 的差别只有 5e-5，所以短音频不用补满 30 秒。
- **Dana 实测**：
  - **数据**：用户录了 5 遍 “Dana”，每遍约 0.8 秒；原模型识别成 Dunner、Gonna…、Donna、GONA!、Donna。用前 4 遍生成 360 条训练样本（16 分钟），第 5 遍只用来验证。
  - **参数**：cactus 的默认参数（lr 1e-4，3 轮）学不会，验证集里第 5 遍还是 Donna。改成 lr 3e-4、5 轮后验证集 15 条对了 14 条，没对上的那一条 Dana 也认对了。这组参数现在是默认值，CPU 上训练 258 秒。
  - **效果**：微调后 5 遍全部识别成 “Dana.”，包括没参与训练的第 5 遍。ALSA 的 8 条测试语音和 test.wav 结果不变。5 条录音倒着放（代替用户说别的话的声音）都没有被识别成 Dana。
  - **改动和取舍**：dummy-message.wav 的转写有小改动，比如 “dummy” 变成 “dime”，“speech this patcher” 变成 “speech vispatcher”。只训练解码器、lr 5e-4 也能认对 Dana，但会把倒放的录音认成 “See you.”，对其他音频改动更大，所以没有用。
- **局限**：只有一个人、一个麦克风、一个房间的 5 遍录音，其他说话人或环境下不一定认得出。“别的话”只有 ALSA 和 speech-dispatcher 的测试音频，覆盖面很窄，不能保证日常说话里完全不误识别。
- **短词更不稳**：1 秒左右的单个词对前后留白很敏感（比如 “Front, Centre.” 加上一点底噪会变成 “FRENT, SENTER.”）；正常长度的句子基本不受影响。
- **一致性检查**：`python scripts/whistle_parity.py --cactus ../cactus` 会用相同的随机权重同时跑原版和移植版，比较参数布局、编码器输出和 logits。完整尺寸下三者都完全一致（差值为 0）。这个检查需要 cactus 的源码，不需要权重。
- **许可证**：cactus 是 Apache-2.0，许可证原文在 `dana/whistle/LICENSE-APACHE`。

### 4.7 语音调用 tools（`dana voice`）

`dana/voice.py` 把两个模型串起来：Whistle（`dana/whistle/listen.py` 的 `heard()`，和 `dana listen` 共用录音、断句和识别）先把每句话转成文字，以唤醒词（默认 “Dana”）开头的句子再交给 Dana 的 `generate()`（`dana/model/run.py`，和 `dana run` 用的是同一个函数），配上 `--tools` 生成调用。`dana/whistle/` 本身仍然不 import Dana 的代码，串联只发生在 `voice.py` 里。

```bash
dana voice --tools examples/tools/smart_home.json
```

- **唤醒词**：取句首的词，去掉标点后不区分大小写比较，“Dana, …” “DANA. …” “Dana …” 都算；剩下的部分首字母大写后作为命令。只说 “Dana.” 时，`--wait`（默认 10 秒）内的下一句是命令。
- **输出**：每条命令在 stdout 输出一行 `{"command": ..., "calls": [...]}`。Dana 输出的不是合法 JSON 时，`calls` 就是 Dana 输出的原文。听到的内容和耗时打印在 stderr。`dana voice` 本身只生成调用，不会执行。
- **实测**：
  - 文字命令（相当于 Whistle 已经认对）配 `smart_home.json`：“Turn on the kitchen lights.”、“Lock the front door.”、“Set the thermostat to 72 degrees.”、“Start the vacuum.” 都生成了正确的调用，“Turn off the bedroom lights and lock the back door.” 生成了两个调用。每条 2.6–5 秒（加 KV cache 之前；现在 0.4–0.6 秒）。
  - 用按实时速度喂入的模拟麦克风测试了四种情况，结果都对：单说 Dana 再说命令；一口气说 “Dana. Front, right.”；没有唤醒词的话；唤醒词等待超时。
  - 还没有用真人说的语音命令测过。
- **控制桌面**：`--tools examples/tools/desktop.json` 有 54 个桌面操作（窗口、应用、音量、媒体、亮度、夜间模式、锁屏和关机、通知、Wi-Fi 和蓝牙、文件夹和文件、网页、打字、快捷键、剪贴板等）。1846 token，超过训练长度，靠上面第 2 节的 YaRN 读完。文字命令实测 51 条里 46 条选对工具，每条约 1 秒（`smart_home.json` 约 0.5 秒）。启动时会打印 tools 的 token 数，超过 4096 时提醒后面的工具被截掉。
- **为什么要唤醒词**：Dana 对任何输入都会给出一个调用，比如 “Front left.” 会变成 `control_blinds(room="front porch", action="close")`。没有唤醒词时，随口说的话也会触发调用。

## 5. 环境变量

| 变量 | 作用 | 不设置时 |
|---|---|---|
| `DANA_DISTRIBUTED` | `1` 或 `0` 强制开/关 `jax.distributed.initialize()` | 只有检测到 Google 的 `/dev/accel0` 时才开 |
| `DANA_ACCELERATOR` | 给 `setup` 指定 `cpu`、`gpu` 或 `tpu` | 自动检测 |
| `GEMINI_API_KEY` | `generate-data` 和 playground 生成数据时用 | — |
| `WANDB_API_KEY` | 配合 `--wandb` 记录实验 | — |

上传类的变量都不设置的话，这个文件夹不会往任何地方上传东西。

## 6. 和上游 `a5af1f8` 相比的改动

完整 diff 见 [docs/local_changes.diff](docs/local_changes.diff)。

- **移除了和 Hugging Face Hub 的所有关联**。上游会从 `Cactus-Compute/needle` 下载权重和分词器（finetune 和 playground 每次启动都强制重新下载），并把 checkpoint、数据集和分词结果上传到 HF。现在全部只读写本地：权重固定用 `checkpoints/dana.pkl`，分词器用 `dana/tokenizer/dana.*`，缺文件直接报错；`DANA_HF_*` 环境变量、`huggingface_hub` 和 `transformers` 依赖都删了；`import dana` 时强制设 `HF_HUB_OFFLINE=1`，`datasets` 库只用来读写本地文件；`pretrain` 必须用 `--data` 指定本地 parquet；`generate-data` 的 `--upload-every` 改名为 `--save-every`；HF 模型卡、HF 格式的 `config.json` 和 HF 下载缓存已删除。
- **`dana tokenize` 不再删除任何东西**。上游会删掉本地数据集和分词器、重新训练分词器，最后还会删掉原始数据。现在只重建 `.data_cache/`，重新训练分词器要显式加 `--train-tokenizer`。
- **`dana generate-data` 改成在本地写 train/validation 两个划分**。上游是合并进私有 HF 数据集，而本地的 `tokenize`/`train` 读不到它写出的格式。
- **`dana pretrain` 新增 `--data`**，可以读本地 parquet：行数从文件里数，只读需要的三列。
- **TPU 检测更严格**。上游只看 `/dev/accel0` 是否存在，Intel 或 AMD 的 NPU 也会被误当成 TPU，导致装 `jax[tpu]`、训练时调用 `jax.distributed.initialize()` 后报错。现在要求 PCI 厂商是 Google。
- **修复 `dana tpu setup/sync` 打包目录的 bug**：上游只打包了 `dana/` 这个包目录，漏掉了 `setup` 和 `pyproject.toml`。
- **新增 `dana/whistle/` 和 `dana transcribe`**：从 cactus 移植的 Whistle 语音识别，独立于 Dana 和 Needle 3 的代码，见 4.6。之后又加了 `dana listen`（麦克风实时识别）、`--keywords`（关键词偏置），以及 `dana record` + `dana finetune-whistle`（录音、LoRA 微调认新词），还有 `dana voice`（语音 → Whistle → Dana → tool 调用，见 4.7）。
- **Dana 推理的输入上限从 1024 提到 4096**：超过训练长度时 RoPE 用 YaRN 缩放，见第 2 节；训练仍然是 1024。**Dana 解码加了 KV cache**（`dana/model/kv_decode.py`），desktop 命令从 5–6.5 秒降到 1 秒左右，结果不变，见第 2 节。`examples/tools/desktop.json` 从 13 个工具扩充到 54 个。
- **项目整体从 needle 重命名为 dana**：包名、命令（`dana ...`）、环境变量（`DANA_*`）、文件名（`checkpoints/dana.pkl`、`dana/tokenizer/dana.*`）都改了。上游 GitHub 链接和论文引用保持原样。

## 7. 注意事项

- `checkpoints/dana.pkl` 是 pickle 文件，加载时可以执行任意代码。这里的文件来自官方仓库，并且已经校验过 sha256。不要加载来路不明的 `.pkl`。
- `checkpoints/`、`dana/tokenizer/`、`dana/data/`、`.data_cache/` 和 `.venv` 都在 `.gitignore` 里。如果 `git init` 这个文件夹，权重不会被提交。
- 这里的 Dana（即 Needle 1）和现在的 `cactus-compute` 包（Needle 3）互不兼容：checkpoint 格式（pkl 对 safetensors）、分词器和工具格式都不一样。
