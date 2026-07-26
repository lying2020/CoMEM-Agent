# CoMEM-Agent：MMInA 最小评估链路复现清单

> 目标：不下载 489 GB CoMEM 轨迹，先跑通 **Qwen2.5-VL-7B Base（无记忆）在 MMInA Wikipedia 上的完整执行与判分链路**。  
> 在线资源核验日期：2026-07-24。

---

# 1. 推荐范围

第一阶段只跑：

```text
MMInA Wikipedia
+ Qwen2.5-VL-7B actor / judge
+ UI-TARS-1.5-7B grounding
+ no memory
```

暂时不需要：

- 489 GB CoMEM trajectories；
- CLIP；
- FAISS；
- CoMEM 17B merged checkpoint；
- Qwen2.5-VL-32B；
- SEAgent；
- Mind2Web / WebVoyager；
- 训练代码、DeepSpeed、FlashAttention-2。

Wikipedia 跑通后再尝试 MMInA Shopping。Shopping 依赖外部 OneStopMarket EC2 服务，稳定性低于 Wikipedia。

---

# 2. 必须下载的模型与数据

## 2.1 Qwen2.5-VL-7B-Instruct

- 作用：
  - 主 GUI Agent；
  - 页面截图描述；
  - 自然语言 action 解析；
  - MMInA fuzzy-match judge；
- Hugging Face：
  - https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- 准确 ID：

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

- 规模：
  - 8.292B 参数；
  - BF16 权重约 16.6 GB；
  - 5 个 safetensors 分片；
  - 连同 tokenizer/processor 约 16.7 GB。
- GPU：
  - BF16 最低约 20–24 GB VRAM；
  - 建议限制 context 和图像数量，避免 24 GB 卡 OOM。

下载示例：

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen2.5-VL-7B-Instruct
```

也可以不预下载，直接让 vLLM 下载到 `HF_HOME`。

## 2.2 UI-TARS-1.5-7B

- 作用：把 click/type/select 的文字描述定位到屏幕坐标；
- Hugging Face：
  - https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B
- 准确 ID：

```text
ByteDance-Seed/UI-TARS-1.5-7B
```

- 规模：
  - 8.292B 参数；
  - 仓库权重约 33.2 GB，HF 标记为 F32；
  - 7 个 safetensors 分片；
  - 以 BF16 加载时，GPU 权重约 16.6 GB，但磁盘仍需存放约 33.2 GB 原始文件。
- GPU：
  - BF16 最低约 20–24 GB VRAM；
  - 48 GB GPU 更稳妥。

下载示例：

```bash
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B \
  --local-dir models/UI-TARS-1.5-7B
```

## 2.3 MMInA 数据集

- Hugging Face：
  - https://huggingface.co/datasets/shulin16/mmina
- GitHub：
  - https://github.com/shulin16/MMInA
- 准确 Dataset ID：

```text
shulin16/mmina
```

- 总规模：
  - 1,050 个任务；
  - 总文件大小约 43.6 MB；
  - Wikipedia：308 条；
  - Shopping：200 条。

下载示例：

```bash
huggingface-cli download shulin16/mmina \
  --repo-type dataset \
  --local-dir data/mmina
```

将任务 JSON 放置或软链接为：

```text
CoMEM-Agent-Inference/
└── mmina/
    ├── wikipedia/
    │   ├── 1.json
    │   ├── 2.json
    │   └── ... 308.json
    └── shopping/
        ├── 1.json
        └── ... 200.json
```

每条 JSON 应包含：

```text
task_id
start_url
intent
eval.eval_types
eval.reference_answers
```

---

# 3. 下载量、磁盘和内存预算

## 3.1 纯 Base MMInA

需要下载：

```text
Qwen2.5-VL-7B       ≈ 16.7 GB
UI-TARS-1.5-7B      ≈ 33.2 GB
MMInA               ≈ 43.6 MB
────────────────────────────
模型 + 数据          ≈ 50 GB
```

还需预留：

- Python/CUDA/vLLM 环境：约 10–25 GB；
- Hugging Face 临时下载和 cache：约 10–30 GB；
- Chromium：约 0.5–1 GB；
- 结果 HTML、base64 截图和日志：
  - 20 条 smoke test：约 0.2–1 GB；
  - Wikipedia 308 条：建议预留 10–30 GB；
  - Wikipedia + Shopping 508 条：建议预留 20–50 GB。

因此：

- **最低可用磁盘**：100 GB 空闲；
- **推荐磁盘**：150 GB 空闲；
- 若模型目录和 HF cache 各保存一份，建议 200 GB。

## 3.2 主机内存

- 最低：64 GB RAM；
- 推荐：128 GB RAM；
- 原因：两个 vLLM 进程、模型加载 staging、Chromium 和截图处理会同时占用 CPU 内存。

## 3.3 如果以后只加 CoMEM checkpoint

不下载轨迹，只下载：

```text
WenyiWU0111/lora_qformer_test_V4-700_merged ≈ 33.5 GB
```

总模型磁盘会从约 50 GB 增加到约 83.5 GB。但**没有轨迹/FAISS 时 CoMEM 没有可检索记忆，不能形成有意义的 CoMEM 评测**，所以第一阶段无需下载它。

---

# 4. GPU 方案

## 4.1 推荐配置

### 方案 A：2×24 GB

```text
GPU 0：Qwen2.5-VL-7B，port 8000
GPU 1：UI-TARS-1.5-7B，port 8001
```

可选卡：

- RTX 4090 24GB；
- RTX 3090 24GB；
- L4 24GB；
- A10 24GB。

要求：

- BF16/FP16；
- `max-model-len` 先限制为 4096 或 8192；
- `gpu-memory-utilization` 约 0.85–0.90；
- 每个 prompt 只允许少量图片。

这是成本最低且最容易调试的配置。

### 方案 B：2×48 GB

例如 L40S、A6000/Ada 48GB。最稳妥，能容纳更长 context 和更高截图 token。

### 方案 C：1×80 GB

例如 A100/H100 80GB。可以尝试在同一 GPU 启动两个 vLLM 服务，但要分别限制显存比例，存在显存碎片和双进程竞争。可用，但不如两卡隔离稳定。

## 4.2 不推荐

- 单张 24 GB：不能同时稳定驻留两个 BF16 7B VLM；
- CPU-only：GUI 每一步需要多次 VLM 调用，速度不可接受；
- Apple Silicon/MPS：vLLM、CUDA 与当前代码链路不匹配。

---

# 5. 软件环境

推荐：

```text
OS              Ubuntu 22.04/24.04 x86_64
Python          3.10
NVIDIA Driver   与所选 CUDA/PyTorch wheel 对齐
CUDA            12.x
PyTorch         CUDA build
vLLM            支持 Qwen2.5-VL 的当前稳定版本
Transformers    >= 4.49；与 vLLM 版本共同锁定
Browser         Playwright Chromium
```

macOS 可用于准备数据和阅读结果，不适合承担本项目的正式本地推理。

基础依赖：

```text
torch
torchvision
transformers
accelerate
vllm
openai
playwright
gymnasium
beartype
numpy
pandas
matplotlib
pillow
requests
aiohttp
beautifulsoup4
qwen-agent
qwen-vl-utils
```

建议 actor/benchmark 环境与 vLLM 服务环境分开：

```text
env-agent：Playwright、评测代码、轻量 Python 包
env-vllm：PyTorch、vLLM、Transformers
```

安装骨架：

```bash
conda create -n comem-agent python=3.10 -y
conda activate comem-agent

pip install openai playwright gymnasium beartype \
  numpy pandas matplotlib pillow requests aiohttp \
  beautifulsoup4 qwen-agent qwen-vl-utils

playwright install chromium
```

vLLM 建议按官方当前 CUDA wheel 或 Docker 安装，不要先固定一个未经验证的旧版本。环境跑通后立即导出：

```bash
pip freeze > requirements.lock.txt
```

---

# 6. 启动前必须修复的代码

这些问题不修，即使模型和数据全部下载完成也无法正常运行。

## 6.1 修复 vLLM client 创建

文件：

```text
CoMEM-Agent-Inference/agent/llm_config.py
```

将：

```python
api_key = args.getattr("open_router_api_key", "EMPTY")
```

改成：

```python
api_key = getattr(args, "open_router_api_key", "EMPTY")
```

## 6.2 关闭默认训练数据收集

文件：

```text
CoMEM-Agent-Inference/config/argument_parser.py
```

当前：

```python
parser.add_argument("--collect_training_data", action="store_true", default=True)
```

应将默认值改为 `False`。否则 Base 评测也会产生训练数据副作用。

## 6.3 支持 MMInA 小样本切片

当前 `run.py` 没有把 `test_start_idx/test_end_idx` 传给 MMInA。

应把：

```python
test_file_list = create_test_file_list_mmina(args.domain)
```

改成：

```python
test_file_list = create_test_file_list_mmina(
    args.domain,
    args.test_start_idx,
    args.test_end_idx,
)
```

否则第一次 smoke test 也会直接尝试整个 Wikipedia 308 条。

## 6.4 建议修复 CLI 布尔参数

将：

```python
parser.add_argument("--use_memory", type=bool, default=False)
```

改为：

```python
parser.add_argument("--use_memory", action="store_true")
```

Base 模式虽然不使用该参数，但应在正式实验前修正。

## 6.5 确认明文凭据已删除

`browser_env/action_parser_ground.py` 中存在明文第三方账号信息。应先轮换和删除，禁止直接用真实账户运行。

---

# 7. 启动两个模型服务

以下使用 Hugging Face ID；如果已下载，可把 `--model` 改成本地绝对路径。

## 7.1 Qwen 服务：GPU 0 / port 8000

```bash
export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

## 7.2 UI-TARS 服务：GPU 1 / port 8001

```bash
export CUDA_VISIBLE_DEVICES=1

vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --trust-remote-code
```

如果 24 GB 卡 OOM：

1. 将 `max-model-len` 降为 4096；
2. 将图片上限降为 1；
3. 将 `gpu-memory-utilization` 调至 0.85；
4. 确认没有其他进程占显存；
5. 再考虑量化，而不是一开始就改变模型精度。

## 7.3 健康检查

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8001/v1/models
```

两个请求都应返回模型列表。

---

# 8. 分阶段运行

## 8.1 Level 0：浏览器测试

确认：

- Chromium 能启动；
- 能访问 Wikipedia；
- 能生成 1280×720 screenshot；
- 无系统代理或证书错误。

## 8.2 Level 1：单条 Wikipedia

完成第 6.3 节的切片修复后，从仓库根目录运行：

```bash
./CoMEM-Agent-Inference/run_baseline.sh \
  --eval_type mmina \
  --domain wikipedia \
  --model qwen2.5-vl \
  --max_steps 15
```

当前 shell 脚本没有转发 `test_start_idx/test_end_idx`。首次测试建议直接调用 Python：

```bash
cd CoMEM-Agent-Inference

python run.py \
  --evaluation_type mmina \
  --domain wikipedia \
  --model qwen2.5-vl \
  --max_steps 15 \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --debug \
  --result_dir results/smoke/mmina-wikipedia-1
```

## 8.3 Level 2：20 条 Wikipedia

```bash
python run.py \
  --evaluation_type mmina \
  --domain wikipedia \
  --model qwen2.5-vl \
  --max_steps 15 \
  --test_start_idx 0 \
  --test_end_idx 20 \
  --debug \
  --result_dir results/smoke/mmina-wikipedia-20
```

## 8.4 Level 3：Wikipedia 全量

```bash
python run.py \
  --evaluation_type mmina \
  --domain wikipedia \
  --model qwen2.5-vl \
  --max_steps 15 \
  --test_start_idx 0 \
  --test_end_idx 308 \
  --result_dir results/mmina/wikipedia/qwen25vl-base
```

## 8.5 Level 4：Shopping

确认以下地址可访问后再运行：

```text
http://ec2-3-146-212-252.us-east-2.compute.amazonaws.com:7770/
```

如果不可访问，Shopping 不能按当前代码完成。不要把网站不可达计为模型失败。

---

# 9. 每个阶段的验收标准

单条任务必须确认：

- `run.py` 能导入全部模块；
- 端口 8000/8001 均可连接；
- 页面正常加载；
- Qwen 能返回合法 action；
- UI-TARS 坐标落在 1280×720 viewport 内；
- click/type 后页面发生变化；
- Agent 最终输出 STOP；
- 结果目录生成 `render_<task_id>.html`；
- 日志出现 PASS 或 FAIL；
- evaluator 没有异常；
- 任务之间状态已清空。

20 条测试需统计：

```text
总任务数
成功数
失败数
blocked / inaccessible 数
early-stop 数
judge 解析失败数
平均动作数
平均每条耗时
```

当前代码没有可靠的统一聚合脚本，需要扫描日志或补一个结果汇总器。

---

# 10. 最终采购清单

## 最低可行

```text
GPU              2×24 GB
CPU RAM          64 GB
SSD 空闲          100 GB
OS               Ubuntu 22.04/24.04
模型             Qwen2.5-VL-7B + UI-TARS-1.5-7B
数据             MMInA 43.6 MB
```

## 推荐

```text
GPU              2×48 GB，或 1×80 GB
CPU RAM          128 GB
SSD 空闲          150–200 GB
OS               Ubuntu 22.04/24.04
```

第一阶段的实际网络下载量约 **50 GB**，不包含 Python/CUDA wheels；完全不需要下载 489 GB CoMEM 轨迹。

---

# 11. Vanilla、文本记忆与 CoMEM 的关系

## 11.1 三组基本对照

论文和代码中的三条路径应区分为：

```text
Vanilla
├── 不使用外部记忆
├── 不使用 Q-Former
├── 不加载 CoMEM LoRA/checkpoint
└── 原始 backbone 直接推理

Text-based Memory
├── 使用外部轨迹库
├── CLIP + FAISS 检索相关历史轨迹
├── 将轨迹动作整理为文本示例并拼入 prompt
├── 不使用 Q-Former
└── 不加载 CoMEM 训练得到的 LoRA/连续记忆编码器

CoMEM
├── 使用外部轨迹库
├── CLIP + FAISS 检索相关历史轨迹
├── 读取历史动作与截图
├── encoder VLM + Q-Former 将每条轨迹压缩为连续向量
├── 连续向量直接插入 backbone 输入 embedding
└── 使用作者训练后的 CoMEM checkpoint
```

因此：

\[
\boxed{
\text{“有记忆、但没有 CoMEM LoRA”}
\approx
\text{Text-based Memory}
}
\]

文本记忆不是“无记忆模型的另一个名字”。它确实检索了历史经验，只是把经验表示为离散文本 token，而不是连续 embedding。

## 11.2 LoRA 的准确含义

CoMEM 中的 LoRA 不是用于把 GUI Agent backbone 普遍微调成更强的 Agent，而是连续记忆编码路径的一部分。

论文把训练概括为：

```text
冻结 backbone VLM
+ LoRA
+ Q-Former
≈ 1.2% 可训练参数
```

但当前代码的实际实现更具体：

- `knowledge_processor`（Q-Former）是完整可训练模块；
- LoRA 加在 encoder VLM 的部分线性层；
- 最终负责动作生成的 `model_inf` 主干保持冻结；
- LoRA rank 为 16；
- 每条历史轨迹由 Q-Former 压缩为固定数量连续向量。

所以更准确的表述是：

\[
\text{Trainable Memory Encoder}
=
\text{full Q-Former}
+
\text{LoRA-adapted encoder VLM}
\]

而不是简单地说“LoRA 加在 Q-Former 上”。

## 11.3 “没有检索到记忆”不是独立 baseline

连续记忆运行时如果没有检索到有效经验，可能退回普通推理路径。但这是运行时 fallback，不是论文中的独立实验组。

论文的标准对照仍是：

1. Vanilla；
2. Text-based Memory；
3. CoMEM。

---

# 12. MMInA 数据集究竟是什么

## 12.1 MMInA 是 GUI 数据集吗

准确地说，MMInA 是一个**在线多模态 Web GUI Agent benchmark**。

它不是：

- ScreenSpot 一类静态截图 grounding 数据集；
- 预先保存完整鼠标动作序列的 imitation-learning 数据集；
- 预先构造好的多轮“用户—助手”图文对话；
- CoMEM 的记忆训练轨迹库。

它由两部分共同构成：

```text
文本任务配置
+
真实/动态网页环境
```

Agent 需要在浏览器中观察网页截图、执行 GUI 动作，并最终完成任务。

## 12.2 下载下来的内容

MMInA 的核心数据是任务 JSON。典型结构如下：

```json
{
  "task_id": 17,
  "sites": ["shopping"],
  "start_url": "https://...",
  "intent": "比较两个城市的信息，然后在目标城市查找活动",
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {
      "must_include": ["eventbrite", "tokyo"]
    }
  },
  "procedure": ["kiwix", "event", "end"]
}
```

主要字段含义：

- `task_id`：任务编号；
- `start_url`：浏览器初始页面；
- `intent`：用户文本任务；
- `sites`：任务涉及的网站；
- `procedure`：多跳任务参考流程；
- `eval.eval_types`：判分方法；
- `eval.reference_answers`：参考答案或必须包含的内容。

数据集本身通常不包含：

- 每一步网页截图；
- 标准鼠标坐标；
- 完整专家动作轨迹；
- 多轮截图—动作 demonstration；
- 已经构建好的 FAISS index。

## 12.3 多模态与多轮从哪里产生

多轮图文交互是在评测运行时动态生成的：

```text
文本任务 + start_url
        ↓
Playwright 打开网页
        ↓
获取当前页面截图
        ↓
Agent 输入：
任务文本 + 当前截图 + 最近动作历史
        ↓
Agent 输出：
click / type / select / scroll / wait / stop
        ↓
UI-TARS 根据截图与元素描述预测坐标
        ↓
Playwright 执行动作
        ↓
网页更新并获取下一张截图
        ↓
循环，直到 STOP、失败或达到 max_steps
```

因此 MMInA 的整体形式是：

\[
\boxed{
\text{静态任务 JSON}
+
\text{在线网页 GUI 环境}
\Longrightarrow
\text{运行时多轮图文轨迹}
}
\]

## 12.4 当前 CoMEM 仓库中的每轮输入

当前实现每一步大致向 Agent 提供：

```text
System Prompt
+ 当前网页截图
+ 模型生成的页面文字描述
+ 最近动作历史
+ 当前任务文本
```

Agent 输出结构化动作，例如：

```json
{
  "name": "click",
  "arguments": {
    "description": "搜索按钮",
    "reasoning": "提交当前查询"
  }
}
```

UI-TARS 再将“搜索按钮”定位为屏幕坐标，Playwright 执行点击。

## 12.5 MMInA 与 CoMEM 轨迹数据的区别

两者不能混为一谈：

```text
MMInA
├── 用途：评测
├── 大小：约 43.6 MB
├── 内容：任务、URL、参考答案、判分规则
└── 不提供完整历史轨迹

CoMEM memory trajectories
├── 用途：记忆检索、训练 memory encoder
├── 大小：约 489 GB
├── 内容：任务、每轮截图、动作、模型响应、成功/失败信息
└── 可用于建立 CLIP + FAISS memory bank
```

本清单的第一阶段只使用 MMInA，不下载 CoMEM memory trajectories，因此能够验证：

- 浏览器环境；
- Qwen Agent；
- UI-TARS grounding；
- 多轮交互；
- MMInA evaluator；
- Base task accuracy。

但不能在这一阶段验证 Text-based Memory 或 CoMEM 的实际增益，因为这两者都需要额外的历史轨迹库。

