# CoMEM-Agent 推理复现数据集需求清单与 Situation Memory 适用评估

> 审计对象：当前仓库、论文 arXiv:2510.09038、作者 README、Hugging Face / GitHub 公开资源  
> 核验日期：2026-07-27  
> 目标：单独梳理**推理评测与记忆构建所用数据集**；给出下载可行性；评估 **Situation Memory** 在各主数据集上的可应用比例与潜在影响程度。  
> 相关文档：`CoMEM-Agent_完整推理复现准备评估.md`（代码/模型阻塞）、`MMInA_最小复现清单.md`（最小链路）、`GUI_Situation_Hypergraph_Memory_研究思路.md`（方法设计）。

---

# 0. 一句话结论

论文主评测只用 **MMInA / Multimodal-Mind2Web / WebVoyager**（仓库有 runner）；**GUI-Odyssey / OSWorld** 是 OOD 表但无代码入口；记忆库是 **CoMEM trajectories（HF 现仓约 455 GiB）**。  
若只做 Situation Memory 可行性摸底：**立即下载 MMInA（≈0.04 GiB）+ WebVoyager JSONL（≈141 KB）**；Multimodal-Mind2Web **全仓约 12.6 GiB**（card 上旧 `download_size≈4 GB` 不可信），请只下测试 split（各自 ≤4 GiB）；GUI-Odyssey 截图 / OSWorld VM / 完整 memory ZIP 均远超 4 GiB。

Situation 证据链建议：**Mind2Web 离线检索正确性 → MMInA 在线因果验证 → GUI-Odyssey 跨 App → OSWorld final-state**。CoMEM 解决「怎么压缩/注入」；Situation 解决「按阶段与约束检索什么」。

---

# 1. 仓库概览（与数据相关）

```text
CoMEM-Agent/
├── README.md                          # 论文入口、benchmark 列表、HF 轨迹与 checkpoint
├── requirements.txt                   # 依赖声明存在（与旧审计文档不一致处以本文件为准）
├── CoMEM-Agent-Inference/             # 推理与在线评测主仓
│   ├── run.py / run_baseline.sh
│   ├── MMInA_evaluation/              # MMInA runner + evaluator
│   ├── Mind2Web_evaluation/           # Mind2Web 在线改造评测
│   ├── webvoyager_evaluation/         # WebVoyager 在线评测
│   ├── mmina/                         # 仅 README，任务 JSON 未入库
│   ├── memory/                        # CLIP + FAISS 经验检索
│   ├── memory_evolution/              # 数据飞轮
│   └── data_preparation/              # Mind2Web/训练数据转换脚本
├── CoMEM-Agent-train/                 # Q-Former / LoRA 训练
└── docs/                              # 论文 PDF、TeX、组会与复现笔记
```

**仓库内已支持的 `eval_type`：** `mmina`、`mind2web`、`webvoyager`。  
**论文有结果但仓库无 runner：** GUI-Odyssey、OSWorld。  
**本地现状：** 三个主 benchmark 的任务配置文件均未随仓发布，需自行下载/转换。

---

# 2. 数据集总表（论文 + 仓库实际用到）

| 角色 | 数据集 | 论文用途 | 本仓可跑？ | 规模（公开口径） | 下载体量（2026-07-27 核验） | ≤4 GiB？ | 建议 |
|------|--------|----------|------------|------------------|------------------------------|----------|------|
| 主评测 | MMInA (Wiki+Shop) | 主表 Task Acc | 有 runner，缺 JSON | 308+200；全集 1050 | **43,600,377 B ≈ 0.041 GiB** | 是 | **立即下载** |
| 主评测 | Multimodal-Mind2Web | 主表 Task Acc（在线改造） | 有 runner；**evaluation_data 已本地转换（836 条）** | 14k actions / 2k+ tasks | **全仓 ≈12.64 GiB**；test_website≈1.03 / test_domain≈3.37 / test_task≈1.19 / train≈7.06 GiB | 测试 split 是；全仓否 | parquet 在 `dataset/multimodal-mind2web/`；在线 JSON 在 `Mind2Web/evaluation_data/` |
| 主评测 | WebVoyager | 主表 Task Acc | 有 runner，缺 per-site JSON | 643 任务 / 15 站 | **≈141 KB** + ref≈103 KB | 是 | **立即下载**；`domain=test` 示例会报错 |
| OOD | GUI-Odyssey（新） | AMS | **无** | 8334 ep / 212 apps | **≈86.46 GiB**；截图分卷 ≥6–10 GiB | 标注是；截图否 | **只下 annotations/splits** |
| OOD | OSWorld | SR | **无** | 369 Ubuntu 任务 | 任务 JSON 很小；file cache≈**1.09 GiB**；VM≈**78.4 GiB** | 任务/cache 是；VM 否 | 先任务定义；VM 延后 |
| 记忆/训练 | CoMEM trajectories | memory bank | 需转换 | 188,451 轨迹 | **≈455.3 GiB**；仅 `services.zip`≈1.81 GiB | 子集是；全仓否 | tasks/links + 抽样；勿整包 |
| 飞轮种子 | Mind2Web train | \(\mathcal Q_0\) | 转换脚本有 | 1009 / 7775 | 见 Multimodal train≈7.06 GiB | 否（整 split） | 建库阶段再下 |
| 遗留未接 | VisualWebArena / WebArena / WebWalkerQA / expand_memory | 非主表 | helper 或注释 | — | — | — | 不纳入第一轮 |

体量以 HF API 文件字节和为准；card 旧 `download_size=4.01GB`（Multimodal-Mind2Web）与现仓 **12.64 GiB** 冲突，**以下载规划用现仓字节**。论文「100k+ / $4k」与 README「188,451 / $1,972」不一致，需记 revision。

---

# 3. 主数据集逐项说明

## 3.1 MMInA

### 评测目的与意义

- **目的：** 在真实网站上的多跳、多模态 Web GUI 任务上，衡量 Agent 的长程规划与跨站执行能力。  
- **论文意义：** 主表核心；Wikipedia / Shopping 上 CoMEM 相对 Base 提升最大（论文：Wiki 36.7→47.4；Shop 15.5→45.0），是「记忆是否有用」的最强证据位。  
- **对 Situation Memory：** 多跳 `procedure`、跨站实体交接（Wiki→Eventbrite 等）天然对应 **Handoff / Phase** 节点，最适合验证「表面不相似、结构相同」的联想检索。

### 评测范围（论文 vs 全集）

| 子集 | 任务数 | 论文是否使用 |
|------|--------|--------------|
| wikipedia | 308 | 是（全量） |
| shopping | 200 | 是（全量） |
| normal / multi567 / compare / multipro | 176+180+100+86 | 否（官方全集共 1050） |

指标：Task Accuracy；论文写 LLM-as-Judge；本仓实现为 `string_match` / `url_match` / `program_html`，**未完整实现官方逐 hop 协议**——汇报时需区分「CoMEM adapter 成绩」与「官方 multihop 成绩」。无 SPDX license；README 标明评测用途。

### 大小与下载

- HF：https://huggingface.co/datasets/shulin16/mmina  
- GitHub：https://github.com/shulin16/MMInA  
- 体量：**43,600,377 bytes ≈ 0.041 GiB**（其中 `teaser.png` 约占 40 MB；任务 JSON 本身很小）  
- **≤4 GiB：是 → 建议现在下载。**

```bash
huggingface-cli download shulin16/mmina \
  --repo-type dataset \
  --local-dir data/mmina
```

放置约定：

```text
CoMEM-Agent-Inference/mmina/{wikipedia,shopping}/*.json
```

### Situation 适用评估

| 维度 | 评估 |
|------|------|
| 可应用任务比例（粗估） | 保守 **50–65%**；中性 **65–80%**（明确多跳子集占全集一半以上；Wiki/Shop 另含比较/约束任务） |
| 主导 situation 类型 | CrossSiteEntityHandoff、HopProgress、CompareThenAct、PendingDependency、NoResultRecovery |
| 预期影响程度 | **3–4 / 高**。论文记忆增益最大；适合做**第一个在线因果**实验 |
| 验证成本 | **低**。数据小、本仓有 runner；Shopping 依赖 OneStopMarket/EC2 |
| 推荐优先级 | **P1 在线验证**（离线机制先用 Mind2Web） |

---

## 3.2 Multimodal-Mind2Web

### 评测目的与意义

- **目的：** 跨网站、跨领域的开放式 Web 任务；官方数据含截图、HTML、专家动作。  
- **论文意义：** 主表按 Shopping / Travel / Info / Service 报 Task Acc；强调未见网站/领域 OOD。注意：论文是**在线整任务 + VLM judge**，不是官方离线 element accuracy。  
- **对 Situation Memory：** 多步表单与多约束任务适合 **ConstraintProgress / Phase**；专家轨迹可自动对齐「阶段标签」做监督与离线诊断。

### 评测范围

| Split | Tasks / Actions（官方） | 论文用法 |
|-------|-------------------------|----------|
| train | 1009 / 7775 | 飞轮种子 / 训练来源之一；非主表测试 |
| test_website | 142 / 1019 | 取前 100，跳过失效站 |
| test_domain | 694 / 4060 | 取前 100，跳过失效站 |
| test_task | 177 / 1339 | 论文未强调 |

精确 allowlist、最终有效分母未在仓库发布；当前 `run.py` 会 glob 全目录，**复现论文数字必须自行截断到前 100 + 失效过滤**。

本仓期望：

```text
CoMEM-Agent-Inference/Mind2Web/evaluation_data/<domain>/**/*.json
# 字段至少: task_id, intent, start_url
```

### 大小与下载

- HF：https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web（公开，`openrail`）  
- **现仓文件合计 ≈ 12.64 GiB**（不要用 card 旧统计 `download_size≈4.01 GB` 做下载预算）  
- Split 实际体积：train≈**7.06 GiB**；test_domain≈**3.37**；test_task≈**1.19**；test_website≈**1.03 GiB**  
- 原始 Mind2Web Raw Dump ≈300 GB，**不要下**  
- **≤4 GiB：三个测试 split 各自满足；全仓 / train 不满足。建议只下目标测试 split。**  
- 在线评测若只要 `intent/start_url`，可从 parquet 导出轻量 JSON（远小于 4 GiB）。  
- 纯文本 Mind2Web：`osunlp/Mind2Web` 全仓≈6.28 GiB；`test.zip`≈0.53 GiB 可单独下。

```bash
# 推荐：仅 test_website（≈1.03 GiB）
huggingface-cli download osunlp/Multimodal-Mind2Web \
  --repo-type dataset \
  --include "data/test_website*" \
  --local-dir data/multimodal-mind2web-test_website

# 需要论文 OOD domain 时再下 test_domain（≈3.37 GiB）
huggingface-cli download osunlp/Multimodal-Mind2Web \
  --repo-type dataset \
  --include "data/test_domain*" \
  --local-dir data/multimodal-mind2web-test_domain
```

### Situation 适用评估

| 维度 | 评估 |
|------|------|
| 可应用任务比例（粗估） | 保守 **60–75%**；中性 **75–90%** |
| 主导 situation 类型 | ConstraintProgress、SearchFilterVerify、FormPhase、TargetDisambiguation |
| 预期影响程度 | 离线 step metric **2–3**；检索研究 **3–4**。标签可观测性最高（截图+HTML+专家动作） |
| 协议注意 | 官方=离线 element/op/step；本仓=在线浏览器+LLM judge——**两套成绩必须分开命名** |
| 推荐优先级 | **P0：首轮离线检索验证** |

---

## 3.3 WebVoyager

### 评测目的与意义

- **目的：** 15 个动态真实网站上的开放问答/操作；强调多模态 grounding 与动态页面。  
- **论文意义：** 主表单列；CoMEM 54.5，相对闭源也有竞争力。采用 WebSight「achievable」子集，**allowlist 未随本仓发布**。  
- **对 Situation Memory：** 多约束搜索、候选核验、冲突约束决策适合 **ConstraintCheck**；但任务常无固定专家轨迹，situation 需在线从截图+指令抽取。

### 评测范围

- 官方：`WebVoyager_data.jsonl`，约 **643** 条（本机拉到 642 行量级，以文件为准）。  
- 论文：achievable subset（数量未在正文固定）。  
- 本仓期望：`webvoyager_evaluation/data/<site>/*.json`（`task_id/intent/site/start_url`）。

### 大小与下载

- https://github.com/MinorJerry/WebVoyager  
- 任务文件：https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/WebVoyager_data.jsonl  
- 体量：约 **141 KB**（另有 `reference_answer.json` ≈100 KB）  
- **≤4GB：是 → 建议下载。**

注意：动态站会过期；2026 年直接跑全量成功率会系统性偏低，需记录日期与失败原因（blocked / 改版 / 登录）。

### Situation 适用评估

| 维度 | 评估 |
|------|------|
| 可应用任务比例（粗估） | 保守 **35–50%**；中性 **50–70%** |
| 主导 situation 类型 | CandidateVerification、PartiallySatisfiedConstraints、SearchRefinement、NoResultRecovery |
| 预期影响程度 | **2–3 / 中**。单站短问答边际价值有限 |
| 推荐优先级 | **P2 补充**；示例脚本 `--domain test` 与目录枚举不一致，需改用真实 site 名 |

---

## 3.4 GUI-Odyssey（OOD）

### 评测目的与意义

- **目的：** 移动端跨 App 导航；专家完整轨迹 + AMS。  
- **论文意义：** Web 记忆/编码器直接测 mobile OOD。CoMEM High AMS 22.38→27.41；Low 略降。支撑「连续记忆高层可迁移、文本记忆易负迁移」。  
- **对 Situation Memory：** **SystemSettingDetour / AppSwitch** 是 situation 联想的最强证据场景之一（Settings→改权限→重开 App）。

### 评测范围

- 新版 `hflqf88888/GUIOdyssey`：**8,334** episodes / 127,893 screenshots / 212 apps（官方推荐）。  
- 旧版 `OpenGVLab/GUI-Odyssey`：7,735 episodes；现仓文件和 ≈**62.28 GiB**。  
- 论文未写清用哪一版；复现需固定 revision。  
- **本仓无 eval_type / AMS 接口。**

### 大小与下载

- 新版：https://huggingface.co/datasets/hflqf88888/GUIOdyssey → 现仓 ≈**86.46 GiB**；截图分卷单卷 6–10 GiB，需齐套才能解压  
- 旧版：https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey  
- GitHub：https://github.com/OpenGVLab/GUI-Odyssey  
- **≤4 GiB：标注与 split manifest 是；截图全量否。现在只建议取标注。**

### Situation 适用评估

| 维度 | 评估 |
|------|------|
| 可应用任务比例（粗估） | 保守 **70–85%**；中性 **85–95%**（结构匹配度最高） |
| 主导 situation 类型 | CrossAppEntityHandoff、SystemSettingDetour、ReturnToSourceApp、BackStackRecovery |
| 预期影响程度 | **3–4**；标签可观测性 4；离线 AMS 无需 Android，在线执行另需 runner |
| 推荐优先级 | **P1：跨 App 泛化证明**；须用 task/app split，忌 random split |

---

## 3.5 OSWorld（OOD）

### 评测目的与意义

- **目的：** 真实桌面 OS 工作流；执行式 final-state evaluator（文件/配置等）。  
- **论文意义：** CoMEM Overall SR 26.40→26.73，增益极小；文本记忆下降。说明「web 记忆直接搬到 desktop」弱迁移。  
- **对 Situation Memory：** 长流程 **Workflow Phase**（Print→PDF→Margins→Save）适合；但需要 VM 与状态观测，situation 特征与 web 差异大。

### 评测范围

- Ubuntu：**369** 任务（可排除 8 个 Google Drive 相关 → 361）；另有 Windows 43 任务。  
- 类别：Office / Daily / Professional / Workflow 等。  
- **本仓无 adapter。**

### 大小与下载

- 项目：https://github.com/xlang-ai/OSWorld（Apache-2.0；369 Ubuntu tasks，可跳过 8 个 Drive → 361）  
- 任务定义：`evaluation_examples/examples`（远低于 4 GiB）  
- file cache：`xlangai/ubuntu_osworld_file_cache` ≈ **1.09 GiB**（可下）  
- VM：`xlangai/ubuntu_osworld` ≈ **78.4 GiB**；单架构 zip ≈11.3–11.4 GiB  
- **≤4 GiB：任务 + file cache 是；VM 否。勿与 gated 的 OSWorld-V2 task classes 混用。**

### Situation 适用评估

| 维度 | 评估 |
|------|------|
| 可应用任务比例（粗估） | 保守 **55–70%**；中性 **70–85%** |
| 主导 situation 类型 | DesktopWorkflowPhase、DialogOrFilePickerStage、ArtifactCreatedPendingVerification |
| 预期影响程度 | **4（本域）**；用 web bank 直接测则接近论文微弱增益。**严禁**把 evaluator/expected result 编入 situation |
| 推荐优先级 | **P2/P3** |

---

## 3.6 CoMEM Memory Trajectories（记忆库，非主 benchmark）

### 评测/使用目的

- **目的：** 构建 FAISS memory bank；抽 1,500 条训 Q-Former；支撑 scaling 实验。  
- **意义：** 飞轮产物本身；不是标准公开 benchmark 分数来源。  
- **对 Situation：** 应用比例取决于轨迹是否含可解析的多阶段结构；成功轨迹约 38.7k / 188k，更适合作 situation 标注池。

### 大小与下载

- https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories（Apache-2.0）  
- 现仓 ≈ **455.3 GiB**（488,914,890,967 bytes）；多数 ZIP 5–150+ GiB  
- ≤4 GiB 可用件：generated tasks / expand links；以及 `expand_memory/.../services.zip` ≈ **1.81 GiB**  
- **全轨迹：否。** 建议先 tasks/links + 极小成功子集建索引。

代码期望目录形态（需转换）：

```text
training_data/<dataset>/<domain>/qwen2.5-vl-32b/<run>/success/*.jsonl
# 实际 json.load 整文件；扩展名 jsonl 易误导
```

### Situation 适用评估

| 维度 | 评估 |
|------|------|
| 可应用比例 | **中：约 40–60%（成功轨迹中）**；失败/未完成轨迹噪声大 |
| 影响 | 作为 **retrieval corpus** 决定 situation 召回上限；比 benchmark 本身更关键「bank 覆盖」 |
| 推荐 | 与 MMInA/Mind2Web 评测并行建 **小而干净** 的 situation-indexed bank |

---

# 4. Situation Memory：跨数据集适用比例与影响总览

> 下列比例是基于任务结构、论文记忆增益、公开样例与本仓研究笔记的**研究向粗估**，不是已跑出的实测覆盖率。落地时应用规则/模型自动标注后重估。

## 4.1 评分定义

- **适用比例：** 任务中至少存在一类可稳定抽取的 situation（Handoff / Phase / ConstraintProgress / ConstraintCheck / AppSwitch / Detour 等）的占比。  
- **影响程度：** 相对「仅 CLIP 任务+首图检索」或「无记忆」，用 Situation 索引后对成功率/AMS/错误类型的**预期改善幅度**（High / Mid / Low）。  
- **与 CoMEM 关系：** Situation 改检索；CoMEM 改表示。组合预期 ≥ 单独任一（需 surface-disjoint 评测，不能只报原表 Acc）。

## 4.2 总览表

| 数据集 | 适用比例（保守→中性） | 影响 | 优先级 | 最敏感指标 |
|--------|------------------------|------|--------|------------|
| Multimodal-Mind2Web | 60–75% → 75–90% | 3–4（检索） | **P0** | Situation Recall@K / step metrics |
| CoMEM traj + Mind2Web train | 35–50%→55–75% / 60–90% | 语料 3 | **P0 建库** | Precision@K、去重泄漏率 |
| MMInA Wiki/Shop | 50–65% → 65–80% | 3–4 | **P1 在线** | Task Acc、跨站失败 |
| GUI-Odyssey | 70–85% → 85–95% | 3–4 | **P1** | High-level AMS |
| WebVoyager | 35–50% → 50–70% | 2–3 | **P2** | 约束冲突任务 Acc |
| OSWorld | 55–70% → 70–85% | 4（本域） | **P2/P3** | SR（禁 evaluator 泄漏） |

## 4.3 影响机制（按失败模式）

1. **跨站/跨 App 交接失败**（MMInA、GUI-Odyssey）  
   - Situation 提供 `acquired_entity + pending_subgoal`；CLIP 易被网页皮肤带偏。  
   - **预期影响：高。**

2. **同任务不同阶段误检索**（Mind2Web、OSWorld）  
   - 任务文本相似但应点的控件不同；需要 Phase / ConstraintProgress。  
   - **预期影响：高。**

3. **候选满足大部分约束但仍冲突**（WebVoyager）  
   - 需要 ConstraintCheck 策略记忆，而非「如何打开搜索框」。  
   - **预期影响：中。**

4. **纯视觉 grounding / 坐标偏移**  
   - Situation 帮助有限；仍依赖 UI-TARS / CoMEM 视觉压缩。  
   - **预期影响：低。**

5. **跨平台 OOD（web→desktop）**  
   - 若不重建 situation bank，影响接近论文 CoMEM 的微弱增益。  
   - **预期影响：低到中低。**

## 4.4 建议实验顺序（数据下载与科研 ROI）

```text
阶段 A（≤1 GiB，立刻可下）
  1. MMInA 全集（≈0.04 GiB）+ WebVoyager JSONL
  2. Multimodal-Mind2Web test_website（≈1.03 GiB）→ 离线 situation 检索
  3. OSWorld 任务定义（可选）+ 勿下 VM

阶段 B（≤4 GiB 增量）
  4. test_domain（≈3.37 GiB）或 CoMEM services.zip（≈1.81 GiB）抽样
  5. MMInA 在线：只换 retriever，固定 bank/top-K/actor
  6. GUI-Odyssey：仅 annotations + split manifests

阶段 C（大体积）
  7. GUI-Odyssey 截图分卷 / OSWorld VM / CoMEM 全轨迹
  8. 仅在需要 AMS 全量或主表数字对齐时进行
```

证据链：**Mind2Web 离线检索 → MMInA 在线增益 → GUI-Odyssey 跨 App → OSWorld final-state。**

---

# 5. 推理复现：数据集侧需求清单（可勾选）

## 5.1 最小可跑（推荐先做）

- [ ] 下载 `shulin16/mmina`，放入 `mmina/wikipedia`（可选 shopping）  
- [ ] 下载 `WebVoyager_data.jsonl`，转换为本仓 per-site JSON（需自写几十行转换）  
- [ ] **不必**下载 CoMEM 全仓（≈455 GiB）、GUI-Odyssey 截图、OSWorld VM  
- [ ] 模型与 GPU：见 `MMInA_最小复现清单.md`（Qwen2.5-VL-7B + UI-TARS 等）

## 5.2 论文主表对齐（数据部分）

- [ ] MMInA：Wiki 308 + Shop 200  
- [ ] Mind2Web：自 HF 转出在线 config；**test_domain / test_website 各前 100 + 失效过滤**；记录最终分母  
- [ ] WebVoyager：取得或重建 WebSight achievable allowlist；固定评测日期  
- [ ] Memory：成功轨迹子集 + 自建 FAISS（作者 index 未发布）  
- [ ] 固定 judge：端口 8000 模型与 prompt revision  

## 5.3 Situation Memory 专用数据工作

- [ ] 在 MMInA 上标注/抽取：`phase, acquired_entity, pending_subgoal, handoff`  
- [ ] 在 Mind2Web 专家步上对齐：`completed_constraints / pending_constraints`  
- [ ] 构造 surface-disjoint pair（同结构、不同站/实体）评估检索，而不只报原 Acc  
- [ ] GUI-Odyssey：统计 AppSwitch / Settings detour 占比（可先用 annotation-only）  
- [ ] 明确负例：单跳无约束任务 → 期望 Situation 与 CLIP 接近（防止「处处加权」）

## 5.4 明确不做或延后

- [ ] 不下载 Mind2Web Raw Dump（≈300 GB）  
- [ ] 不整包下载 GUI-Odyssey（≈93 GB）/ OSWorld VM（≈84 GB+）除非进入阶段 C  
- [ ] 不把「能下载」等同于「能复现论文数字」：缺 allowlist、FAISS、失效站列表与 judge 版本

---

# 6. 下载命令速查

```bash
# 1) MMInA ≈ 0.041 GiB  ✅
huggingface-cli download shulin16/mmina --repo-type dataset --local-dir data/mmina

# 2) WebVoyager ≈ 141 KB  ✅
curl -L -o data/WebVoyager_data.jsonl \
  https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/WebVoyager_data.jsonl

# 3) Multimodal-Mind2Web：按 split，勿下全仓 12.64 GiB
huggingface-cli download osunlp/Multimodal-Mind2Web --repo-type dataset \
  --include "data/test_website*" --local-dir data/mm_mind2web_test_website   # ≈1.03 GiB ✅

# 4) GUI-Odyssey：只标注  ✅；截图分卷 ❌
# huggingface-cli download hflqf88888/GUIOdyssey --repo-type dataset \
#   --include "all_annot.json" "annotations/*" "*_split*" --local-dir data/guiodyssey-ann

# 5) OSWorld：任务 + 可选 file_cache≈1.09 GiB  ✅；VM≈78 GiB ❌
# https://github.com/xlang-ai/OSWorld  → evaluation_examples/
# https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache

# 6) CoMEM：全仓 ≈455 GiB ❌；services.zip≈1.81 GiB 可选
# https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories
```

---

# 7. 与 Situation 研究的决策建议

1. **数据投入优先序：** Mind2Web test（离线）> MMInA（在线）> GUI-Odyssey 标注 > WebVoyager > OSWorld 任务 > 大轨迹/VM。  
2. **Situation 最值得做的三个：** Mind2Web（约束进度，P0）、MMInA（交接，P1 在线）、GUI-Odyssey（跨 App，P1）。  
3. **影响预期：** Mind2Web/MMInA 追求可测检索与成功率；OSWorld+web bank 勿期望 Shopping 级跃升。  
4. **下载策略：** ≤4 GiB 的 MMInA + WebVoyager + Mind2Web **单个 test split** 足够第一轮；**不要**按旧 card 去下「4GB 全量 Mind2Web」。

---

# 8. 口径与风险备注

- 体量以 2026-07-27 HF/GitHub **文件字节和**为准；Multimodal-Mind2Web card 的 `download_size≈4.01GB` 与现仓 12.64 GiB 冲突，已废弃作预算。  
- 遗留未接入：`VisualWebArena` / `WebArena` / `WebWalkerQA` / `supergpqa` / `expand_memory`（`run_baseline.sh` 未接）；WebQA 仅经 MMInA wikipedia 108 题间接引用。  
- 在线 benchmark 可复现性受网站改版影响大于文件完整性。  
- 代码阻塞见 `CoMEM-Agent_完整推理复现准备评估.md`。  
- Situation 比例为结构估算，标注 100+ 任务后应用实测覆盖率替换。
