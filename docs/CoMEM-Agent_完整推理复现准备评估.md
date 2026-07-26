# CoMEM-Agent 完整推理与验证复现准备评估

> 审计对象：当前本地仓库代码、论文 PDF/TeX、作者 README、Hugging Face 与 benchmark 官方资源  
> 在线资源核验日期：2026-07-24  
> 目标：从一台新机器出发，跑通 Base / Text-Memory / Continuous-Memory，并尽可能复现论文主实验。

---

# 0. 最终判断

## 0.1 当前仓库能否直接运行

**不能。当前代码不是开箱即用的完整复现包。**

Python 文件和 shell 文件在语法层面可以解析，但从入口启动会遇到确定性阻塞：

1. 没有 `requirements.txt`、`pyproject.toml` 或 `environment.yml`；
2. 三个主 benchmark 的任务配置均未包含在仓库中；
3. 推理 prompt 已存在，但训练数据准备脚本仍引用不存在的旧路径 `GUI-Agent/agent/prompts/system_prompt_simple.txt`；
4. 没有发布在仓库中的 FAISS memory index；
5. 默认 `training_data/` 记忆轨迹目录缺失；
6. `args.getattr(...)` 会在创建任何 vLLM client 时直接报错；
7. continuous-memory 示例传入的 `--checkpoint_path` 不被 shell 和 Python 参数解析器接受；
8. continuous-memory 示例脚本的反斜杠与注释会切断命令；
9. 代码硬编码了模型端口、项目路径、网站 URL 和服务器路径；
10. MMInA、Mind2Web、WebVoyager 的论文精确任务子集/失效任务列表没有发布在当前仓库；
11. 代码没有 GUI-Odyssey 和 OSWorld 的执行/评分入口；
12. 代码中存在明文账号凭据，必须立即删除并轮换，不能在真实账号上运行。

因此要区分三个目标：

- **跑通一个任务**：修代码、下载少量任务和模型即可；
- **复现主表的 CoMEM 结果**：还需要论文同版 memory index、精确 task allowlist、judge 配置和网站快照；
- **复现完整数据飞轮**：还需约 489 GB 发布数据或自行 rollout、Qwen2.5-VL-32B、SEAgent、SerpAPI、浏览器集群和安全治理。

## 0.2 推荐目标

第一阶段不要下载完整 489 GB 数据，也不要直接跑 100k 轨迹。建议先完成：

1. MMInA Wikipedia 5–20 条；
2. Qwen2.5-VL-7B Base；
3. 同一任务集上的 Text-Memory；
4. 作者 Qwen CoMEM checkpoint + top-3 memory；
5. 固定 judge 后比较 task success、耗时、动作数和显存。

---

# 1. 实际运行架构

## 1.1 Base / Text-Memory

代码默认需要两个 OpenAI-compatible vLLM 服务：

```text
port 8000: Qwen/Qwen2.5-VL-7B-Instruct
           ├── 主 Agent
           ├── 页面描述模型
           ├── action 文本解析器
           ├── content analyzer
           └── benchmark judge

port 8001: ByteDance-Seed/UI-TARS-1.5-7B
           └── click/type/select 坐标 grounding
```

Text-Memory 还需：

```text
轨迹 JSON + 截图
   ↓
openai/clip-vit-base-patch32
   ↓
FAISS IndexFlatIP
   ↓
top-k 轨迹被转成文字动作示例，写入 system prompt
```

## 1.2 Continuous-Memory

除上述两个服务外，还会在当前 Python 进程加载：

```text
WenyiWU0111/lora_qformer_test_V4-700_merged
   ├── encoder Qwen2.5-VL
   ├── 8-query Q-Former
   └── frozen model_inf Qwen2.5-VL
```

Hugging Face 将该 checkpoint 识别为约 **17B 参数**，不是普通 7B adapter。原因是 checkpoint 实际包含 encoder、推理模型及 Q-Former 等权重。它不能当作几百 MB 的 LoRA 直接挂载到 7B vLLM 上。

连续记忆的数据流：

```text
任务 + 第一张截图
   ↓ CLIP
FAISS top-k
   ↓
读取被检索轨迹的多张截图与动作
   ↓ encoder VLM
Q-Former：每轨迹 8 vectors
   ↓ 当前实现截断/补齐为总计 24 vectors
prepend 到当前 Qwen 输入 embedding
   ↓
生成下一动作
```

当前 Agent 在每个任务第一次生成 system prompt 时检索一次，后续步骤复用相同经验；并不是每一步基于新截图重新检索。

---

# 2. 必需的开源模型与在线状态

## 2.1 完成最小 Base 推理必须下载/部署

### Qwen2.5-VL-7B

- Repo ID：`Qwen/Qwen2.5-VL-7B-Instruct`
- URL：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- 状态：公开、非 gated；
- Hugging Face 标注：约 8B 参数，BF16；
- 用途：主 Agent、工具模型、默认 judge；
- 建议：vLLM 以 BF16 启动，端口 8000；
- 官方提醒：Transformers 必须支持 `qwen2_5_vl`，并安装 `qwen-vl-utils`。

### UI-TARS grounding

- Repo ID：`ByteDance-Seed/UI-TARS-1.5-7B`
- URL：https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B
- 状态：公开；
- Hugging Face 标注：约 8B；
- 用途：所有 click/type/select 的坐标 grounding；
- 建议：单独 vLLM 服务，端口 8001。

即使跑 Qwen Base，当前 `run.py` 也会无条件创建 UI-TARS client，因此 8001 服务不可缺省，除非修改代码允许直接使用模型坐标。

## 2.2 完成 Continuous-Memory 推理必须下载

### Qwen2.5-VL + CoMEM

- Repo ID：`WenyiWU0111/lora_qformer_test_V4-700_merged`
- URL：https://huggingface.co/WenyiWU0111/lora_qformer_test_V4-700_merged
- 状态：公开、非 gated；
- 模型卡：没有；
- Hugging Face 标注：约 17B 参数，BF16/F16，分片 safetensors；
- Inference Provider：无；
- 用途：Qwen2.5-VL-7B 的 continuous-memory 本地推理；
- 风险：没有训练数据版本、commit、运行参数、期望 Transformers 版本说明。

### UI-TARS + CoMEM

- Repo ID：`WenyiWU0111/lora_qformer_uitars_test_V1-400_merged`
- URL：https://huggingface.co/WenyiWU0111/lora_qformer_uitars_test_V1-400_merged
- 状态：公开；
- 模型卡：没有；
- Hugging Face 标注：约 17B 参数；
- 用途：UI-TARS backbone 的 CoMEM 实验；
- 注意：当前代码映射存在歧义；启用 continuous memory 后参数解析器把 `args.model` 强制改为 `agent-qformer`，不会自然选择 UI-TARS checkpoint，需要额外参数和代码修正。

## 2.3 检索模型

### CLIP

- Repo ID：`openai/clip-vit-base-patch32`
- URL：https://huggingface.co/openai/clip-vit-base-patch32
- 状态：公开；
- 用途：文本/首截图 embedding，建立 FAISS key；
- 代码默认 cache 路径写成 `/.cache/huggingface/hub`，普通用户可能没有根目录写权限，建议改为用户 cache。

## 2.4 完整数据飞轮才需要

### Qwen2.5-VL-32B rollout actor

- Repo ID：`Qwen/Qwen2.5-VL-32B-Instruct`
- URL：https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
- 状态：公开；
- Hugging Face 标注：约 33B；
- 用途：论文中自动轨迹 rollout；
- 端口：代码把 `qwen2.5-vl-32b` 固定到 8004；
- 资源：BF16 权重约 66 GB，实际 vLLM 需额外 KV cache 和视觉激活，建议至少 1×H100/H200 80GB，或 2×48GB/80GB tensor parallel。

### SEAgent trajectory judge

- 准确 Repo ID：`Zery/SEAgent-1.0-7B`
- URL：https://huggingface.co/Zery/SEAgent-1.0-7B
- 状态：公开；
- Hugging Face 标注：约 8B，BF16；
- 用途：论文数据飞轮的轨迹质量筛选；
- 说明：当前仓库的主推理/benchmark evaluator 没有实际加载该模型；如复现数据飞轮，需要按 SEAgent 官方仓库单独接入。

## 2.5 Baseline 可选模型

### CogAgent

- Repo ID：`zai-org/cogagent-9b-20241220`
- URL：https://huggingface.co/zai-org/cogagent-9b-20241220
- 状态：公开；
- Hugging Face 权重：约 13.9B tensors、27.8 GB，带自定义 ChatGLM 代码；
- 许可证：`other`，使用前需单独核对许可；
- 注意：该模型不是标准对话模型，要求专用输入拼接和字符串动作格式；
- 当前仓库只是把它接到通用 OpenAI chat wrapper，未实现官方输入/输出适配，不能认为配置模型名后即可公平复现。

### GLM-4.1V

- Repo ID：`zai-org/GLM-4.1V-9B-Thinking`
- URL：https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking
- 状态：公开；
- 当前仓库 `model_name_map` 没有 GLM 条目，论文表中有结果但代码不能直接启动该 baseline。

### WebSight

- 代码错误地写成：`WenyiWU0111//websight-7B_combined`；
- 可核验的 GUI Agent checkpoint 是：`tanvirb/websight-7B`；
- URL：https://huggingface.co/tanvirb/websight-7B
- 状态：公开，约 8B、F16，基于 UI-TARS-1.5-7B；
- 该仓库缺少部分常见 tokenizer/processor 文件，可能需要从 UI-TARS 基座加载 processor；
- `HuggingFaceM4/WebSight` 是 screenshot-to-HTML 数据集，不是这里的 GUI Agent；
- 因此需要修正 model map 并补 processor/action adapter，当前代码仍不能直接复现 WebSight baseline。

### 闭源模型

代码通过 OpenRouter 使用：

- `openai/gpt-4o`
- `anthropic/claude-sonnet-4`
- `google/gemini-2.5-pro`

需要 OpenRouter key。论文表中的名称/版本与当前代码映射并不完全一致，2026 年 API snapshot 与论文时期也会不同，不能直接声称严格复现原表。

---

# 3. Memory 数据与索引

## 3.1 作者发布的轨迹数据

- Dataset ID：`WenyiWU0111/CoMEM-agent-memory-trajectories`
- URL：https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories
- 状态：公开；
- 总文件大小：约 **489 GB**；
- Expanded links：35,000+；
- Generated tasks：222,235；
- 总轨迹：188,451；
- 成功轨迹：38,731；
- admitted failure：10,394；
- incomplete failure：125,450；
- other failure：13,876；
- 数据卡成本：\$1,972。

论文 PDF 的口径是 100k+ 轨迹、10k+ 环境、约 \$4,000；这是版本差异，复现时必须记录下载日期和 dataset revision。

## 3.2 是否必须下载全部 489 GB

不建议。

推理 memory bank 只应使用：

- 成功轨迹；
- 目标实验需要的 domain；
- 符合代码目录约定的 Qwen2.5-VL-32B rollout；
- 轮数在代码过滤范围内；
- 任务描述与截图完整。

建议先选择 100–1,000 条成功轨迹构建小索引，确认代码链路；再扩到 5k、10k，最后才测试大 bank。

## 3.3 FAISS index

当前仓库没有可直接加载的：

```text
*.faiss
*.embeddings.npy
*.json
```

作者 checkpoint 的一些配置暴露了作者集群上的绝对 FAISS 路径，但不是可下载索引。在线检索也未确认作者公开了与论文主表完全一致的 index revision。

因此必须二选一：

1. 向作者索取论文主表使用的 FAISS 三件套和 trajectory file mapping；
2. 从发布轨迹自行重建，并接受结果不一定与论文一致。

自行建库时当前代码使用：

\[
k_i=[\mathrm{CLIPText}(q_i);\mathrm{CLIPImage}(I_{i,1})],
\]

L2 归一化后写入 `faiss.IndexFlatIP`。

## 3.4 当前数据目录格式要求

`Memory._load_all_conversations()` 期待近似：

```text
training_data/
└── <dataset>/
    └── <domain>/
        └── qwen2.5-vl-32b/
            └── <run-name>/
                └── success/
                    └── *.jsonl
```

但文件扩展名虽为 `.jsonl`，代码实际使用 `json.load()`，即每个文件必须是一个完整 JSON object，而不是标准逐行 JSONL。

每个文件至少要有：

```text
task_description
total_rounds
rounds[]
  ├── response
  └── messages[].content[].image_url.url
```

发布数据是否完全符合这一目录和字段约定，需要在下载样本后做转换验收，不能只把 Hugging Face dataset 根目录直接传给 `--memory_data_dir`。

---

# 4. Benchmark 数据、运行方式和判分

## 4.1 MMInA

### 官方资源

- HF：https://huggingface.co/datasets/shulin16/mmina
- GitHub：https://github.com/shulin16/MMInA
- 总计 1,050 任务；
- 本论文主表只使用 Wikipedia 与 Shopping；
- 论文使用这两个 domain 的全量 308 + 200 = 508 条，任务选择逻辑与当前代码一致；
- 当前代码期待：

```text
CoMEM-Agent-Inference/mmina/wikipedia/1.json ... 308.json
CoMEM-Agent-Inference/mmina/shopping/1.json ... 200.json
```

本地只有 `mmina/README.md`，没有任务 JSON。

### 当前代码判分

根据每条任务的 `eval.eval_types` 组合：

1. `string_match`
   - exact match；
   - must include；
   - fuzzy match 使用端口 8000 的 Qwen2.5-VL 作 yes/no judge；
2. `url_match`
   - exact URL 或 gold URL 是预测 URL 子串；
3. `program_html`
   - 读取页面 HTML / JavaScript 值并检查目标内容；
4. 多 evaluator 分数相乘，全部满足才成功。

这比论文中“最终答案与 ground truth 交给 LLM”更复杂。评测结果取决于任务 JSON 中的 eval 配置。

### 环境风险

- Shopping URL 被硬重定向到一个 EC2 OneStopMarket 服务，当前是否长期可用无法保证；
- Wikipedia 起始页被统一替换为公网 Wikipedia；
- 含登录任务需要账号，但 `ACCOUNTS={}`；
- 动态站点、验证码、地区限制会导致不可比；
- 论文使用的可访问任务集合没有版本化。

## 4.2 Multimodal-Mind2Web

### 官方资源

- HF：https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web
- train：1,009 tasks / 7,775 actions；
- test_task：177 tasks / 1,339 actions；
- test_website：142 tasks / 1,019 actions；
- test_domain：694 tasks / 4,060 actions；
- 数据集约 22 GB 解压规模，下载约 4 GB。

论文进行的是在线任务级成功评测，不是官方离线 element accuracy。

### 当前代码要求

```text
CoMEM-Agent-Inference/Mind2Web/evaluation_data/<domain>/**/*.json
```

仓库没有该目录，也没有完整脚本把 HF 的 action-level offline dataset 转成这里要求的在线 config：

```text
task_id
intent
start_url
```

现有 `mind2web_training_data_conversion.py` 是训练数据转换，不是论文在线评测 split 的完整重建器。

### 当前判分

- 从渲染结果 HTML 中取最后 5 张截图；
- 提取 `finished(answer=...)`；
- 使用端口 8000 的 Qwen2.5-VL 判断 `<result>SUCCESS</result>`；
- Google start URL 任务被代码直接跳过；
- blocked/about:blank 任务直接返回，但未形成统一 denominator 记录。

论文称使用 test-domain 和 test-website 的前 100 条并跳过失效网站；当前 `run.py` 会 glob 全目录，没有前 100 限制。因此必须得到作者精确转换后任务目录或重新实现相同筛选。

## 4.3 WebVoyager

### 官方资源

- 官方项目：https://github.com/MinorJerry/WebVoyager
- 原始任务文件：https://github.com/MinorJerry/WebVoyager/blob/main/data/WebVoyager_data.jsonl
- 原始 benchmark：643 tasks / 15 sites；
- 动态网站任务会过期，2026 年许多实现会更新日期或剔除不可完成任务。

论文采用 WebSight 的 achievable subset，但没有在当前仓库发布精确 allowlist。

### 当前代码要求

```text
CoMEM-Agent-Inference/webvoyager_evaluation/data/<site>/*.json
```

本地没有该目录。

官方 JSONL 字段为 `web_name/id/ques/web`，而本仓库需要
`task_id/intent/site/start_url` 并按站点拆成单个 JSON；当前没有提供这个转换脚本。

### 当前判分

与 Mind2Web 相同：

- 读取 render HTML；
- 取最后 5 张截图和 final answer；
- Qwen2.5-VL judge 输出 SUCCESS / NOT SUCCESS。

这不是 WebVoyager 原论文的固定 GPT-4V snapshot。若要对齐本文，必须固定当前论文使用的 judge checkpoint/prompt，而当前论文未给 judge 精确模型。

## 4.4 GUI-Odyssey 与 OSWorld

论文有 OOD 结果，但当前仓库：

- 没有 dataset 下载脚本；
- 没有环境适配；
- 没有 evaluator；
- `run.py` 不支持这两个 evaluation type。

因此不能用当前仓库复现论文 OOD 表。需要另行集成官方 benchmark，并定义如何把 web memory 输入移动/桌面 actor。

公开来源：

- GUI-Odyssey：https://github.com/OpenGVLab/GUI-Odyssey
- GUI-Odyssey 数据：https://huggingface.co/datasets/hflqf88888/GUIOdyssey
- OSWorld：https://github.com/xlang-ai/OSWorld

---

# 5. 当前代码的确定性阻塞与风险

## 5.1 启动即失败

### `args.getattr` 错误

`agent/llm_config.py` 使用：

```python
args.getattr('open_router_api_key', 'EMPTY')
```

标准 `argparse.Namespace` 没有 `getattr` 方法。应使用：

```python
getattr(args, "open_router_api_key", "EMPTY")
```

该问题影响 actor、grounding 和 judge client，任何模式都会失败。

### Prompt 状态与旧路径问题

主推理所需文件实际存在：

```text
agent/prompts/examples.txt
agent/prompts/system_prompt.txt
```

因此 Base/Memory 推理不会因这两个 prompt 缺失而失败。但训练数据准备脚本还读取：

```text
GUI-Agent/agent/prompts/system_prompt_simple.txt
```

这个旧项目路径及文件不存在；如运行 `prepare_training_data_onfly.py`，仍会失败。严格复现时还应固定现有 prompt 的 Git revision。

### Benchmark 配置缺失

MMInA、Mind2Web、WebVoyager 所需任务 JSON 均缺失。

## 5.2 Continuous-Memory 专有阻塞

1. `argument_parser.py` 未声明 `--checkpoint_path`；
2. `run_baseline.sh` 未解析/转发 `--checkpoint_path`；
3. `run_baseline.sh` 未转发 `--faiss_index_path`、`--similar_num`、`--bank_size`；
4. continuous-memory 示例脚本被行尾反斜杠后的注释切断；
5. `DirectTransformersModel` 硬编码 Python path：`CoMEM-Agent/CoMEM-Agent-train`；
6. 本地真实目录关系与该相对路径不一致时导入失败；
7. checkpoint 自定义类依赖本地 `qwenVL_inference.py`，不能只靠 `AutoModel` 加载；
8. checkpoint 无模型卡，依赖版本不明确；
9. 当前实现把 memory token 强制成 24，与论文 K=10/50/100 实验不一致。

## 5.3 Memory 专有阻塞

1. `training_data/` 缺失；
2. FAISS index 缺失；
3. 如果 `training_data/` 为空，`Memory` 无法建立有效 index，后续保存时仍访问 `self.faiss_index.ntotal`；
4. CLIP cache 写到了根目录 `/.cache/...`；
5. Text-Memory 模式也强制 `multimodal=True`，并不是纯文本检索；
6. 无论文主表对应的 bank revision 和轨迹 ID；
7. IndexFlatIP 对 38k 成功轨迹尚可，但更大 bank 是精确 \(O(Nd)\) 搜索，需评估延迟。

## 5.4 参数和脚本问题

1. `--use_memory` 与 `--use_continuous_memory` 使用 `type=bool`，`"False"` 也会被解析为真；
2. `collect_training_data` 是 `store_true` 但默认值为 `True`，实际上无法通过 CLI 关闭；
3. `save_examples_memory` 同样默认 `True`；
4. continuous memory 会把 `args.model` 强制改为 `agent-qformer`，丢失用户选择；
5. README 示例参数名与实际 shell/Python 参数不完全一致；
6. `run_baseline.sh` 根据当前 cwd 执行 `cd CoMEM-Agent-Inference`，必须从仓库根运行；
7. `run_chunks.py` 共享对象与浏览器并发安全性没有充分保证；
8. 没有 vLLM health check，服务未启动时会在运行中才失败。

## 5.5 模型适配问题

1. CogAgent、WebSight、GLM 与通用 Qwen chat/action prompt 不同；
2. 仅提供 model ID 和端口映射不足以公平运行；
3. UI-TARS 自身输出坐标格式与当前 Agent JSON 工具格式需要适配；
4. 闭源模型通过 OpenRouter，但代码会打印 API key，存在泄漏风险；
5. evaluator、页面描述、action parser 默认复用 Qwen2.5-VL-7B，论文未完整披露这一点。

## 5.6 安全阻塞

`browser_env/action_parser_ground.py` 中存在明文第三方账号和密码。必须：

1. 立即轮换对应凭据；
2. 从 Git 历史中清理；
3. 改为环境变量或受控 secret manager；
4. 禁止在真实账号、支付、私有数据和写操作页面上运行；
5. 对所有自动点击增加域名 allowlist 和 destructive-action denylist。

报告不复述具体凭据值。

---

# 6. 需要安装的软件依赖

当前仓库没有锁定文件。按 imports 至少需要：

```text
Python 3.10+
torch
torchvision
transformers（支持 Qwen2.5-VL 自定义 forward 的兼容版本）
accelerate
peft
deepspeed
flash-attn
liger-kernel
qwen-vl-utils
qwen-agent
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
faiss-cpu 或 faiss-gpu
tqdm
ujson
streaming / mosaicml-streaming
crawl4ai（只用于数据飞轮）
```

浏览器：

```bash
playwright install chromium
```

建议锁定：

- NVIDIA driver；
- CUDA；
- PyTorch；
- Transformers；
- vLLM；
- FlashAttention；
- Playwright/Chromium；
- Qwen checkpoint revision；
- CoMEM checkpoint revision；
- dataset revision。

不要盲目安装“latest”：自定义 monkey patch 和 Qwen generation API 很容易因 Transformers 版本变化失效。

---

# 7. 硬件与存储预算

## 7.1 最小 Base 验证

需要同时提供 Qwen2.5-VL-7B 与 UI-TARS-7B。

推荐：

- 2×24 GB：两个模型分别量化或降低视觉 token / KV cache；
- 1×48 GB：需要严格限制 vLLM `gpu-memory-utilization`，仍可能紧张；
- 2×48 GB 或 1×80 GB：更稳妥。

## 7.2 Continuous-Memory 验证

大致权重组成：

- CoMEM merged checkpoint：约 17B；
- Qwen tool/judge server：约 8B；
- UI-TARS grounding：约 8B；
- 合计约 33B 参数。

BF16 权重本身约 66 GB，还未包含 KV cache、视觉激活、FAISS/CLIP 与 CUDA workspace。

推荐：

- **最低实验配置**：2×80 GB，分开部署 CoMEM 与两个服务；
- **更稳妥**：3×80 GB，CoMEM / Qwen / UI-TARS 各占一组 GPU；
- 只有 1×80 GB 时，需要串行卸载、量化或合并服务，不能按当前代码拓扑直接稳定运行。

## 7.3 完整数据飞轮

额外需要：

- Qwen2.5-VL-32B：1×80 GB 临界，建议多卡；
- SEAgent-1.0-7B；
- 浏览器 worker；
- 489 GB 数据下载空间；
- 解压、索引、render HTML、trace 和缓存空间。

建议至少预留：

- 轨迹数据：0.5 TB；
- 解压/转换副本：0.5 TB；
- 模型与 HF cache：0.15–0.3 TB；
- 结果、截图、trace：0.2 TB；
- 总计约 **1.5 TB SSD** 起步，完整保留中间产物建议 2 TB+。

---

# 8. 修复后建议的运行顺序

## Phase 0：安全和版本冻结

1. 轮换并移除明文凭据；
2. 建立独立 Python/CUDA 环境；
3. 生成依赖锁文件；
4. 记录所有 Git/HF revision；
5. 禁止真实账户、支付和破坏性写操作。

## Phase 1：代码 smoke test

修复：

- `args.getattr`；
- 训练准备脚本的旧 prompt 路径；
- CLI boolean；
- checkpoint/index 参数传递；
- hardcoded path；
- continuous-memory shell；
- collect 默认值。

然后做：

1. import 全部核心模块；
2. 启动浏览器并截图；
3. Qwen 返回一个合法 JSON action；
4. UI-TARS 返回有效坐标；
5. 在自建静态 HTML 页面执行 click/type；
6. evaluator 能读 render HTML。

## Phase 2：Base Agent

启动两个服务（示意，参数需按硬件调整）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --dtype bfloat16
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model ByteDance-Seed/UI-TARS-1.5-7B \
  --served-model-name ByteDance-Seed/UI-TARS-1.5-7B \
  --port 8001 \
  --dtype bfloat16
```

从仓库根目录运行修复后的脚本：

```bash
./CoMEM-Agent-Inference/run_baseline.sh \
  --eval_type mmina \
  --domain wikipedia \
  --model qwen2.5-vl \
  --max_steps 15
```

先只放 1–5 个 MMInA JSON，并检查：

- render HTML；
- 最后答案；
- action 是否实际执行；
- evaluator；
- PASS/FAIL；
- 轨迹耗时。

## Phase 3：Text-Memory

1. 下载 100–1,000 条成功轨迹；
2. 转成代码要求的目录/JSON 格式；
3. 建立 CLIP + FAISS index；
4. 人工检查 top-3；
5. 用固定任务集跑 Base vs Text-Memory。

必须保存：

```text
index.faiss
embeddings.npy
memory metadata JSON
轨迹文件 revision/ID 清单
```

## Phase 4：Continuous-Memory

1. 下载 17B merged checkpoint；
2. 验证自定义类能加载；
3. 单条轨迹压缩形状为 `[1, 8, 3584]`；
4. top-3 合计 24 token；
5. 无浏览器先做一次离线 next-action generation；
6. 再接 MMInA 5 条；
7. 与 Base/Text 使用完全相同任务和 judge。

## Phase 5：主表验证

在三个 benchmark 上固定：

- 任务 ID 清单；
- 失效任务排除清单；
- memory bank revision；
- top-k；
- bank size；
- max steps；
- viewport 1280×720；
- actor temperature/top-p；
- grounding model；
- judge checkpoint/prompt；
- 网站访问日期；
- 随机种子。

至少报告 3 次运行或任务级 bootstrap 置信区间。

---

# 9. 验证方法与验收指标

## 9.1 功能正确性

- 浏览器能打开任务 start URL；
- screenshot 是有效 PNG base64；
- Qwen 输出可解析 action；
- grounding 坐标在 viewport 内；
- action 导致预期页面变化；
- STOP answer 被 render；
- evaluator 能稳定返回 0/1；
- 每个新任务清空上一个任务的 memory cache。

注意：当前 `reset()` 是空实现，虽然 runner 手动清部分 memory 字段，但其他 Agent 内部状态也应系统清理。

## 9.2 Memory 正确性

- FAISS 条目数与 metadata 数一致；
- embedding 已 L2 normalize；
- 查询和 bank 维度一致；
- 不检索自身任务；
- top-k 文件存在且截图可解码；
- 每条经验压缩为 8 token；
- 总前缀长度符合设置；
- 对随机/不相关 memory 做负对照。

## 9.3 论文指标

每个 benchmark 至少报告：

```text
有效任务数
成功任务数
Task Accuracy / Success Rate
平均动作数
Early-stop 比例
Blocked / inaccessible 比例
平均 trajectory wall-clock
检索耗时
轨迹压缩耗时
Agent prefill/decode 耗时
峰值 GPU 显存
judge 失败/不可解析比例
```

当前 runner 没有可靠的统一 aggregate 脚本：`save_scores_to_json()` 虽被定义/导入但未实际调用。需要从日志与 `render_*.html` 汇总，并明确 blocked/skip 是否进入分母；否则无法与论文 Table 2 对齐。

## 9.4 三组核心对照

1. Qwen2.5-VL-7B Base；
2. Base + Text-Memory；
3. Base + CoMEM。

保持 actor、grounding、task、网页时间窗口和 judge 完全相同。否则结果差异不能归因于 memory。

建议额外加：

- random top-k memory；
- text-only retrieval；
- image-only retrieval；
- multimodal retrieval；
- top-k = 1/3/10；
- bank size = 100/1k/5k/10k。

---

# 10. 要严格复现论文主结果，仍需向作者索取

当前公开材料不能确定以下内容：

1. Table 2 使用的精确 task ID / allowlist；
2. Mind2Web 转换后的在线 evaluation configs；
3. WebVoyager achievable subset；
4. 主实验最终有效样本分母；
5. 主实验 top-k；
6. 主实验 memory bank size；
7. 对应 FAISS index 与 metadata；
8. 1,500 条训练轨迹 ID；
9. checkpoint 的精确训练配置与选择标准；
10. Transformers、PyTorch、CUDA、FlashAttention revision；
11. judge 模型准确 ID、revision 与 decoding 参数；
12. Table 4 latency 子集及为何准确率与 Table 2 不一致；
13. UI-TARS baseline 正文 6.6% 与表中 13.2% 的差异；
14. OOD GUI-Odyssey/OSWorld 适配代码；
15. 论文采用的 WebSight 是否确为 `tanvirb/websight-7B`，以及对应 processor/action adapter；
16. 已预建的 memory index 是否可公开。

如果拿不到这些资源，能做的是“基于公开代码的独立复现”，而不是“逐数字复现论文”。

---

# 11. 建议建立的最终可复现目录

```text
CoMEM-Agent/
├── requirements.lock.txt
├── configs/
│   ├── models.yaml
│   ├── services.yaml
│   ├── eval_mmina.yaml
│   ├── eval_mind2web.yaml
│   └── eval_webvoyager.yaml
├── assets/
│   ├── prompts/                # 固定现有 prompt revision
│   ├── benchmark_tasks/
│   ├── memory_trajectories/
│   └── memory_index/
├── manifests/
│   ├── hf_revisions.json
│   ├── task_allowlists.json
│   ├── excluded_tasks.json
│   └── trajectory_ids.json
├── scripts/
│   ├── start_qwen.sh
│   ├── start_uitars.sh
│   ├── build_index.py
│   ├── smoke_test.py
│   └── evaluate.py
└── results/
    └── <date>-<git-sha>-<config-hash>/
```

每次实验保存：

- git SHA；
- HF revisions；
- 完整 CLI/config；
- task IDs；
- 网页访问时间；
- 每步 screenshot/action；
- judge 原始输出；
- aggregate metrics。

---

# 12. 最小采购/准备清单

## 必需

- Linux + NVIDIA GPU 环境；
- 推荐 2×80 GB GPU 做 CoMEM；
- 1.5 TB 以上 SSD（完整数据）；
- Qwen2.5-VL-7B；
- UI-TARS-1.5-7B；
- Qwen CoMEM merged checkpoint；
- CLIP ViT-B/32；
- MMInA/Mind2Web/WebVoyager 任务配置；
- 成功 memory 轨迹子集；
- 自建或作者 FAISS index；
- 已版本化的推理 prompt，以及训练准备旧路径修复；
- 修复后的入口参数；
- 固定 judge。

## 仅完整飞轮需要

- Qwen2.5-VL-32B；
- SEAgent-1.0-7B；
- SerpAPI key；
- Playwright worker 集群；
- 网站 allowlist；
- PII/版权/注入安全过滤；
- 轨迹人工抽检。

## 可选 baseline

- CogAgent；
- GLM-4.1V-9B-Thinking；
- `tanvirb/websight-7B` 及对应 processor/action adapter；
- OpenRouter key 与闭源 API。

---

# 13. 结论

开源模型权重和大规模轨迹数据基本存在，真正缺失的不是“一个模型”，而是把实验闭合起来的工程与复现资产：

\[
\boxed{
\text{依赖锁定}
+\text{prompt/action schema 版本}
+\text{任务配置/allowlist}
+\text{memory index}
+\text{参数修复}
+\text{judge revision}
+\text{安全治理}
}
\]

完成这些之前，README 中的一行命令不能代表论文结果可复现。最现实的路线是先修通 MMInA 小样本三组对照，再向作者索取 exact index 与 task manifest，最后扩展到完整主表。

