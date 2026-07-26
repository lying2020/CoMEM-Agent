# Auto-scaling Continuous Memory for GUI Agent：组会分析汇报

> 论文：Wenyi Wu et al., *Auto-scaling Continuous Memory for GUI Agent*  
> 版本：arXiv:2510.09038v1（2025-10-10），论文源码标为 ICLR 2026  
> 分析依据：论文 PDF、`docs/tex/main.tex`、仓库训练与推理代码  
> 汇报原则：论文没有给出的信息明确标为“未报告”；论文与仓库不一致处单独列出。

---

## 0. 一句话结论

本文不是重新训练一个更大的 GUI Agent，而是为冻结的 VLM Agent 增加一个“可检索的连续经验库”：把一条包含多张截图和动作的长轨迹压成 8 个连续向量，推理时检索相似轨迹、压缩并前缀注入当前 VLM；同时用自动数据飞轮扩充经验库。核心经验结论是：连续记忆随记忆库规模和检索条数增大而持续获益，而文本记忆在检索约 10 条后开始退化。

论文最强结果是 Qwen2.5-VL-7B + CoMEM：

- MMInA：Wiki 47.4%，Shopping 45.0%；
- Mind2Web：Shopping 22.2%，Travel 18.8%，Info 26.5%，Service 17.7%；
- WebVoyager：54.5%；
- 表中总平均为 31.7%。

需要谨慎看待“scaling law”：论文展示的是有限范围内的经验拟合趋势，不是具有外推保证的理论尺度律。

---

# 1. 任务、问题、动机与现有方法局限

## 1.1 任务定义

本文研究纯视觉、截图驱动的 GUI Agent。给定自然语言任务，例如：

> “Find a beginner's acrylic paint set on Amazon, with at least 24 colors, suitable for canvas painting, and priced under \$40.”

Agent 在时刻 \(t\) 只观察当前 GUI 截图：

\[
o_t=\langle I_t\rangle,
\]

并从动作空间中选择下一步动作：

\[
\mathcal A=\{\textsc{CLICK},\textsc{TYPE},\textsc{SCROLL},
\textsc{WAIT},\textsc{STOP}\}.
\]

环境执行 \(a_t\) 后返回下一张截图，形成轨迹：

\[
\tau=\{(o_t,a_t)\}_{t=1}^{T},\qquad T\le T_{\max}.
\]

本文要解决的不是单步元素定位，而是：

1. 长时程、多步骤任务中的规划与约束保持；
2. 面对未见网站、布局、图标和功能时的分布外泛化；
3. 在有限上下文和可接受推理成本下复用大量历史 GUI 经验。

引入记忆后，策略写成：

\[
a_t\sim\pi_\theta(a\mid o_t,m_t),
\]

其中 \(m_t\) 是检索到的历史轨迹经压缩后拼接成的连续向量。

## 1.2 为什么 GUI Agent 特别需要记忆

长时程 GUI 任务具有三个特点：

1. **部分可观测**：当前截图不能表达此前做过什么、隐藏状态是什么。
2. **视觉细节决定动作**：按钮尺寸、位置、图标、价格标签等细节直接决定 grounding 和选择。
3. **误差累积**：前面一次错误点击或约束遗漏会使后续规划整体失效。

人类会调用过去的程序性经验，例如“购物时先列约束，再比较候选，最后验证”。普通 VLM Agent 常在每个任务上重新摸索，导致重复试错和长链错误。

## 1.3 现有方法及局限

### A. 不使用外部记忆的通用 VLM

代表：GPT-4o、Gemini-Pro-Vision、Claude-4、Qwen2.5-VL、GLM-4.1V。

局限：依赖参数内部知识和当前截图；对交互式网页动作分布、长程规划和工具调用适配不足；在未见网站和布局变化下容易失败。

### B. GUI 专项微调模型

代表：UI-TARS-1.5、CogAgent、WebSight。

局限：把经验写入参数，更新成本高；容易受训练域限制；新网站、新流程到来后不能低成本即时写入知识；本文实验中不同任务域表现不稳定。

### C. 文本离散记忆 / 文本 RAG

代表：把历史轨迹概括成文字、工作流或动作说明后拼入 prompt；相关方向包括 AWM、Memp 等。

局限有三层：

1. **长度问题**：一条典型 GUI 轨迹约 10 个“截图—动作”对；论文估算每对约 1,500 token，因此原始轨迹常超过 15,000 token。
2. **信息损失**：把截图转成文字会丢失位置、大小、图标形态和视觉关系。
3. **噪声与注意力竞争**：检索条数增加时，长文本会占据上下文并引入不相关动作模板。论文观察到文本记忆在约 10 条之后性能下降。

### D. 直接拼接原始多模态轨迹

优点是视觉信息完整，但上下文、视觉 token、注意力和显存成本随轨迹数快速增长，不能把几十到上百条经验直接放入模型。

## 1.4 用论文案例解释现有方法为什么失败

任务要求同时满足四个条件：初学者适用、丙烯颜料、至少 24 色、适合画布且价格低于 \$40。

无记忆的 Qwen2.5-VL-7B 过早选择首个商品，遗漏价格等约束。文本记忆可能告诉模型“比较价格”，但未必保留页面中价格标签和颜色数量出现在哪里。

CoMEM 检索过去相似购物轨迹，并把其中截图、查询和动作压成连续向量。模型由这些经验得到一种隐式程序：

\[
\text{解析约束}\rightarrow\text{搜索}\rightarrow
\text{比较候选}\rightarrow\text{逐项验证}\rightarrow\text{选择}.
\]

论文案例显示，加入 CoMEM 后 Agent 会比较多个商品并验证全部约束，而不是选择首项。

这里必须注意：连续向量究竟编码了多少“位置和尺寸”等细节，论文没有给出可解释性探针；目前主要由下游成功率间接支持。

---

# 2. 方法为何这样设计：从动机到算法工具

## 2.1 设计推导

1. 长轨迹太长 \(\Rightarrow\) 固定长度压缩；
2. 文本摘要丢视觉细节 \(\Rightarrow\) 保留截图和动作的多模态表征；
3. 希望不改 Agent 主干 \(\Rightarrow\) 在 embedding 层前缀注入连续向量；
4. 记忆库很大 \(\Rightarrow\) CLIP 表征 + FAISS top-\(k\) 检索；
5. 记忆越多越好但人工轨迹昂贵 \(\Rightarrow\) 自动数据飞轮；
6. 不希望全参训练 \(\Rightarrow\) 冻结 VLM，只训练 Q-Former/LoRA。

## 2.2 整体算法树

```text
Auto-scaling Continuous Memory for GUI Agent
├── A. 记忆生产：Auto-scaling Data Flywheel
│   ├── A1. 环境发现
│   │   ├── 从 Mind2Web 种子任务池 Q0 采样查询
│   │   ├── SerpAPI / DuckDuckGo 返回每条查询 top-20 网站
│   │   └── 可访问性过滤、低质量过滤、URL 去重 → E*
│   ├── A2. 任务生成
│   │   ├── Playwright 打开页面并截图
│   │   ├── VLM 提取页面内容与结构
│   │   ├── 同类 5 个种子任务作为示例
│   │   ├── 每个页面生成 10 个多步、可验证任务
│   │   └── 第二次 VLM 调用重写任务，去除操作提示
│   ├── A3. 轨迹 rollout
│   │   ├── Qwen2.5-VL-32B 作为 actor
│   │   ├── ReAct + tool calling 与网页交互
│   │   └── 收集 τ=(截图, 动作)1:T
│   └── A4. 质量检查
│       ├── SEAgent-1.0-7B 输入任务与完整轨迹
│       ├── 判断任务是否完成
│       └── 仅成功轨迹写回 Q、E、T，进入下一轮
│
├── B. 记忆存储与读取：Retrieval Memory Bank
│   ├── B1. 存储对象：成功轨迹、任务、环境、截图、动作
│   ├── B2. 检索键
│   │   ├── CLIP 编码任务文本
│   │   ├── CLIP 编码截图
│   │   ├── 文本/图像向量拼接并 L2 归一化
│   │   └── 一条轨迹池化为一个 multimodal key
│   ├── B3. 索引：FAISS IndexFlatIP
│   └── B4. 查询：余弦相似度 top-k
│
├── C. 记忆压缩：Continuous Memory Encoder
│   ├── C1. 冻结的 Qwen2.5-VL 编码检索轨迹
│   ├── C2. 取最后层多模态 hidden states
│   ├── C3. 8 个可学习 latent queries
│   ├── C4. 8 层共享参数的 Perceiver/Q-Former
│   │   ├── latent 对轨迹 hidden states 做 cross-attention
│   │   └── residual attention + FFN
│   └── C5. 每条长轨迹 → 8×3584 连续向量
│
└── D. 记忆消费：Memory-augmented Agent
    ├── D1. 将 top-k 轨迹各自压缩
    ├── D2. 拼接连续记忆并前置到当前输入 embedding
    ├── D3. 冻结 Agent 对 [memory; current observation; instruction] 自注意
    ├── D4. ReAct 生成 reasoning + tool call
    ├── D5. SOM 标签定位元素；无效时 UI-TARS grounding 回退
    └── D6. 执行动作，获得新截图，循环至 STOP / 成功 / Tmax
```

## 2.3 三个核心模块如何交互

1. 数据飞轮向记忆库持续“写入”新的成功轨迹。
2. 检索器根据当前任务与截图从记忆库“选择”少量相关轨迹。
3. 记忆编码器把这些轨迹“压缩”为 VLM 可直接使用的连续前缀。
4. Actor 读取连续前缀和当前截图，生成下一动作。

飞轮扩大的是可用经验覆盖面；FAISS 控制每次读取范围；Q-Former控制每条经验的输入长度；冻结 VLM 负责实际决策。

## 2.4 连续记忆编码器的内部公式

设检索轨迹 \(\tau_i\) 经冻结 VLM 编码后得到：

\[
H_i=[h_1,\ldots,h_{L_i}]\in\mathbb R^{L_i\times d},
\qquad d=3584.
\]

Q-Former 维护 \(r=8\) 个可学习 latent：

\[
Z^{(0)}\in\mathbb R^{8\times d}.
\]

代码中的每一层以 latent 为 query，以轨迹 hidden states 与 latent 的拼接为 key/value：

\[
Q=W_Q\operatorname{LN}(Z),\quad
[K,V]=W_{KV}[\operatorname{LN}(H_i);\operatorname{LN}(Z)].
\]

\[
\operatorname{Attn}(Z,H_i)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
\]

残差更新为：

\[
\widetilde Z^{(\ell)}
=Z^{(\ell-1)}+\operatorname{Attn}(Z^{(\ell-1)},H_i),
\]

\[
Z^{(\ell)}
=\widetilde Z^{(\ell)}+\operatorname{FFN}(\widetilde Z^{(\ell)}).
\]

8 次共享参数迭代后：

\[
M_i=\operatorname{LN}(W_oZ^{(8)})\in\mathbb R^{8\times3584}.
\]

对 top-\(k\) 条轨迹：

\[
m_t=[M_{i_1};M_{i_2};\ldots;M_{i_k}],
\]

再与当前输入 embedding 拼接：

\[
X_t^{\mathrm{aug}}=[m_t;X(o_t,q,\text{history})].
\]

Actor 在冻结参数下生成动作：

\[
\mathcal L
=-\sum_j\log p_\theta(y_j\mid y_{<j},X_t^{\mathrm{aug}}).
\]

训练时只对 assistant 动作响应位置计算交叉熵；prompt 部分 label 被设为 \(-100\)。梯度更新 Q-Former 全参数及 encoder VLM 上的 LoRA 参数，推理用的 `model_inf` 主干保持冻结。论文把它概括为“LoRA on Q-Former”，但代码实际将 `knowledge_processor`（Q-Former）作为完整可训练模块保存，并把 LoRA 加到排除 `model_inf` 与 `knowledge_processor` 后的 encoder 线性层；二者表述并不完全一致。

一个重要实现细节：论文概念上每条轨迹压成 8 个向量；当前 Qwen 代码把 3 条检索轨迹合成 24 个向量，并硬截断/补齐至 24。论文的 \(k=10,50,100\) scaling 实验如何与这一代码路径对应，仓库没有完整说明。

---

# 3. 逐步算法流程：用一个例子走完

下面使用论文购物案例：

> 在 Amazon 找一套适合初学者、至少 24 色、适合画布、低于 \$40 的丙烯颜料。

## 3.1 离线阶段：先建立可增长的记忆

### Step 1：初始化三个池

\[
\mathcal Q_0=\text{Mind2Web train queries},\quad
\mathcal E_0=\varnothing,\quad
\mathcal T_0=\varnothing.
\]

\(\mathcal Q\) 是任务池，\(\mathcal E\) 是网站/应用池，\(\mathcal T\) 是成功轨迹池。

### Step 2：发现环境

从 Shopping 类任务中采样“搜索多约束商品”等查询，经 SerpAPI 搜索 top-20 网站；检查 403、404、验证码和稳定性，规范化 URL 并去重，得到新环境 \(\mathcal E^\*\)。

附录给出的实际初始发现结果是 6,676 个 unique page links，覆盖 13 类：

`education, tech, entertainment, travel, health, news, services, shopping, social, food, academic, government, finance`。

### Step 3：生成任务

Playwright 打开网页并截图。VLM 先提取网站用途、导航、交互元素和内容类型，再结合 5 个同类 Mind2Web 示例，为页面生成 10 个：

- 多步骤；
- 可执行；
- 可度量验证；
- 不泄漏点击路径；
- 相互独立

的任务。之后再次调用 VLM，把过度具体的“点击 X”重写为自然用户目标。

### Step 4：rollout

Qwen2.5-VL-32B 根据当前截图做 ReAct 推理，调用 `TYPE/CLICK/SCROLL/WAIT/STOP` 等工具，形成：

\[
\tau_q^\*=\{(I_1,a_1),\ldots,(I_T,a_T)\}.
\]

例如一条成功经验可能是：

```text
TYPE “beginner acrylic paint set 24 colors canvas”
CLICK 某候选商品
检查价格与颜色数
返回结果页并比较另一候选
确认全部约束
STOP
```

### Step 5：验证并写回

SEAgent-1.0-7B 输入任务 \(q\) 和完整轨迹 \(\tau_q^\*\)，判断是否成功。只保留正例：

\[
(\mathcal Q_t,\mathcal E_t,\mathcal T_t)
\rightarrow
(\mathcal Q_{t+1},\mathcal E_{t+1},\mathcal T_{t+1}).
\]

闭环重复运行，即：

\[
\text{discover}\rightarrow\text{generate}\rightarrow
\text{rollout}\rightarrow\text{verify}\rightarrow\text{discover}.
\]

## 3.2 建索引阶段：轨迹如何成为可检索记忆

对每条成功轨迹提取任务文本及首张/相关截图。当前仓库采用 `openai/clip-vit-base-patch32`：

\[
e_i^{text}=f_T(q_i),\qquad e_i^{img}=f_I(I_i).
\]

代码不是加权和，而是直接拼接：

\[
k_i=[e_i^{text};e_i^{img}].
\]

L2 归一化后写入 `FAISS IndexFlatIP`。由于向量已归一化，内积等价于余弦相似度。

仓库还进行轨迹过滤：

- 只读取 Qwen2.5-VL-32B rollout；
- 总轮数 \(3\le T<15\)；
- 任务描述非空；
- 以 `dataset_domain: task` 去重；
- 构造实际经验时删除连续重复动作；
- 动作数达到 10 时隔步下采样。

这些实现细节在论文正文中没有完整披露，但对复现结果有直接影响。

## 3.3 在线阶段：当前任务如何调用记忆

### Step 1：构造查询键

当前任务和当前截图编码为：

\[
k_t=[f_T(q);f_I(I_t)].
\]

### Step 2：top-\(k\) 检索

\[
\mathcal N_k(t)
=\operatorname{TopK}_{i}\;
\frac{k_t^\top k_i}{\lVert k_t\rVert\lVert k_i\rVert}.
\]

例如检索到三条经验：

1. 搜索限定价格的画材；
2. 比较多属性电子产品；
3. 验证商品颜色数量与用途。

论文主张这种检索不要求网站完全相同，而是复用高层程序和视觉—动作模式。

### Step 3：每条轨迹压成 8 个向量

假设每条原轨迹约 \(15{,}000\) token，三条原始拼接约 \(45{,}000\) token。Q-Former 将每条压为 8 个 \(3584\) 维向量：

\[
\tau_i:\ \mathbb R^{L_i\times3584}
\longrightarrow M_i\in\mathbb R^{8\times3584}.
\]

三条经验共 24 个连续向量，而不是约 45,000 个离散/视觉 token。按论文的粗略长度口径，压缩比约为：

\[
\frac{45,000}{24}\approx1875:1.
\]

这只是输入位置数的近似比较，不等同于端到端 FLOPs 或显存压缩比；论文未报告严格 FLOPs。

### Step 4：前缀注入与决策

\[
X_t^{aug}=
[M_1;M_2;M_3;X_{\text{task}};X_{I_t}].
\]

VLM 的自注意可以让当前截图 token 访问记忆向量。Agent 首先解析四项约束，然后输出类似：

```json
{
  "name": "type",
  "arguments": {
    "description": "search box",
    "reasoning": "Search with all key constraints instead of selecting the first item."
  }
}
```

### Step 5：grounding 和环境反馈

系统用 SOM 给当前截图的可交互元素编号。模型必须描述目标并给出标签；若标签无效，则回退到 UI-TARS-1.5-7B 做 grounding。动作执行后产生 \(I_{t+1}\)，继续：

\[
a_{t+1}\sim\pi_\theta(a\mid I_{t+1},m_{t+1}).
\]

直到任务成功、模型 `STOP` 或达到默认 \(T_{\max}=15\)。

## 3.4 训练阶段如何使 8 个向量“对 Agent 有用”

论文使用 1,500 条高质量轨迹；将轨迹的每一步拆成训练实例，并为每个实例检索 top-3 相关经验。训练目标不是重建原轨迹，而是让给定记忆和当前观察时的下一动作生成概率最大：

\[
\min_{\phi,\Delta\theta_{\text{LoRA}}}
-\sum_{(x,y)}\log p_{\theta_0+\Delta\theta}
\left(y\mid x,C_\phi(\tau_{i_1:i_3})\right).
\]

其中：

- \(\theta_0\)：冻结 VLM 主干；
- \(C_\phi\)：Q-Former 压缩器；
- \(\Delta\theta_{\text{LoRA}}\)：低秩适配参数；
- LoRA rank = 16；
- 更新参数约占 1.2%。

因此压缩器学到的不是通用重建表示，而是“对下一步 GUI 动作有用”的任务条件表示。

---

# 4. 数据集、baseline 与成本

## 4.1 记忆构建数据

论文报告：

- 种子任务：Mind2Web training set；
- 自动轨迹：100k+；
- 环境：10k+；
- 采集成本：约 \$4,000；
- 自动生成数据同时具有 step-wise annotation 和 fully automatic annotation；
- 实际训练只抽取 1,500 条高质量轨迹。

仓库 README 更新为：

- 188,451 条轨迹；
- 采集成本 \$1,972。

这与 PDF 的“100k+ / \$4k”不一致，可能是论文提交后数据扩充或成本重新核算。组会汇报实验结论时应以 PDF 版本为准；下载数据和预算规划时应注明仓库最新口径。

## 4.2 主评测数据集

### MMInA

- 完整基准含 1,050 个真实网站任务；
- 本文只评 Wikipedia 与 Shopping；
- 仓库说明分别为 308 和 200 个任务；
- 要求多模态 grounding 与长时程规划；
- 指标为任务准确率。

### Multimodal-Mind2Web

- 2,000+ 开放任务；
- 137 个网站、31 个领域；
- 本文从 `test-domain` 和 `test-website` 子集各取前 100 条，并跳过已失效网站；
- 主表按 Shopping、Travel、Info、Service 报告；
- 重点衡量未见网站/领域上的 OOD 泛化。

“各取前 100 条”和“按四领域汇报”的精确映射、最终有效样本数未完整报告，因此难以只从论文还原每列分母。

### WebVoyager

- 15 个动态真实网站；
- 按 WebSight 使用“可完成任务”子集；
- 具体任务清单和最终样本数未在论文正文给出；
- 仓库含网站域脚本，但论文结果所用 snapshot/allowlist 未版本化。

## 4.3 OOD 评测数据集

### GUI-Odyssey

- 移动端跨应用导航；
- 使用 web 环境建立的记忆和 web 数据训练的编码器直接测试；
- 指标为 AMS；
- 报 High Level 与 Low Level。

### OSWorld

- 桌面操作系统真实工作流；
- 指标为任务成功率 SR；
- 报 Office、Daily、Professional 和 Overall。

## 4.4 每个数据集的具体样例：输入、输出与内部动作

### 4.4.0 阅读约定

下面严格区分两类内容：

1. **真实公开样本**：直接来自数据集 JSON、Hugging Face Dataset Viewer 或官方文档；
2. **解释性 rollout**：为了说明 Agent 如何执行任务而写出的可能动作序列。若原数据集没有提供标准动作轨迹，不能把解释性 rollout 当作 ground truth。

还要区分三种“输出”：

- **答案型输出**：例如网页检索任务最终返回一段文字；
- **动作型输出**：例如 Mind2Web 每一步的 `CLICK/TYPE/SELECT`；
- **环境状态型输出**：例如 OSWorld 最终生成一个 PDF，evaluator 检查文件，而不是比较一句文本答案。

---

### 4.4.1 CoMEM 自动扩展数据与 memory trajectory

#### 数据用途

这是 CoMEM 的记忆构建/训练数据，不是主 benchmark。它包含：

- 搜索得到的扩展 URL；
- 自动生成的任务；
- Agent rollout 得到的多轮截图—动作轨迹；
- world-state model 对轨迹成功、失败和首个错误步骤的判断。

公开来源：

- `WenyiWU0111/CoMEM-agent-memory-trajectories`
- https://huggingface.co/datasets/WenyiWU0111/CoMEM-agent-memory-trajectories

#### 真实生成任务样例

以下是公开文件 `generated_tasks/tasks_services_V1.json` 的第一条真实记录：

```json
{
  "task_description": "Calculate the maximum rent affordable for a household earning $5,000 monthly in San Francisco",
  "expected_outcome": "Should display the calculated maximum rent amount",
  "difficulty": "Easy",
  "category": "services",
  "url": "https://www.redfin.com/how-much-rent-can-i-afford",
  "source_url": "https://www.redfin.com/how-much-rent-can-i-afford"
}
```

它表达的是：

- 输入：月收入 \$5,000、城市 San Francisco；
- 环境：Redfin rent-affordability calculator；
- 目标输出：页面显示 calculator 计算出的 maximum affordable rent。

#### rollout 文件的完整逻辑结构

作者将每条 rollout 保存成一个完整 JSON object，扩展名虽然是 `.jsonl`，但代码使用 `json.load()` 读取：

```json
{
  "session_id": "session_<timestamp>",
  "conversation_id": "<domain>_<task_id>",
  "task_description": "Calculate the maximum rent affordable for a household earning $5,000 monthly in San Francisco",
  "total_rounds": 4,
  "rounds": [
    {
      "timestamp": "<ISO time>",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": {
                "url": "data:image/png;base64,<当前网页截图>"
              }
            },
            {
              "type": "text",
              "text": "<任务、页面描述和动作历史>"
            }
          ]
        }
      ],
      "response": "<本轮结构化动作>"
    }
  ],
  "evaluation": {
    "analysis": "<world-state model 的轨迹分析>",
    "evaluation": {
      "Correctness": true,
      "Redundant": [],
      "Optimized": true,
      "First_Error_Step": null,
      "Error_Type": "",
      "Correct_Action": ""
    }
  }
}
```

其中 `<base64 screenshot>` 是二进制截图的文本编码，无法在汇报中逐字展开，但它是输入的一部分，不是占位文本 token。

#### 内部动作如何产生

一个合理的解释性 rollout 是：

```text
Round 1
输入：任务文本 + Redfin 初始截图
动作：TYPE(text="$5,000", field_description="monthly household income")

Round 2
输入：收入已填写后的截图 + 上一步动作历史
动作：CLICK(description="calculate or affordability result button")

Round 3
输入：计算结果页面截图
动作：STOP(answer="<页面显示的 maximum affordable rent>")
```

注意：上述三步是根据真实任务构造的**解释性执行过程**，不是从 489 GB 压缩包中逐字抽出的某条 ground-truth trajectory。真实 archive 保存的是 Agent 实际生成的动作，可能包含额外 click、scroll、retry。

#### 从公开压缩包中核验的真实 rollout

下面这条记录实际来自作者公开的 `entertainment.zip`：

```text
entertainment/success/
entertainment_tasks_entertainment_v4_400_0.jsonl
```

真实顶层字段：

```json
{
  "session_id": "session_1758084329",
  "session_start": "2025-09-17T08:45:29.700146",
  "conversation_id": "entertainment_tasks_entertainment_v4_400",
  "conversation_start": "2025-09-18T02:31:08.928682",
  "conversation_end": "2025-09-18T02:31:35.108442",
  "task_description": "Please extract the names and follower counts of the top 16 users listed on the page.",
  "total_rounds": 1
}
```

该轮输入包含 “Top Brazil” TikTok chart 的真实网页截图，以及由当前页面生成的文本描述。页面已经直接展示目标信息，因此完整动作序列只有一个 `STOP`：

```json
{
  "name": "stop",
  "arguments": {
    "answer": "The top 16 users and their follower counts are:\n1. TikTok (26.6M)\n2. Ariana Mancuso (23.5M)\n3. Whindersson Nunes (19.7M)\n4. Jessica Kayane (17.9M)\n5. Maiara (17.8M)\n6. Jimone Mendes (14M)\n7. Ritt (13.8M)\n8. Luisa Sonza (12.8M)\n9. Jojo (11.5M)\n10. Anitta (11.5M)\n11. Pabllo Vittar (8.8M)\n12. Nego Ventura (5.5M)\n13. Dan Santang (4.3M)\n14. Bruna Concalves (2.6M)\n15. Leo Santana (1.8M)\n16. Giovanna Lancelotti (1.4M)",
    "reasoning": "The required information is directly visible on the page, and no further actions are needed."
  }
}
```

真实 judge 输出：

```json
{
  "evaluation": {
    "Correctness": true,
    "Redundant": [],
    "Optimized": true,
    "First_Error_Step": "None",
    "Error_Type": "",
    "Correct_Action": "\"\"}</res_dict>"
  },
  "analysis": "The agent successfully extracted the required information from the visible page content."
}
```

`Correct_Action` 中残留的 `\"\"}</res_dict>` 是原始生成/解析残留，应原样视为数据质量问题，而不能擅自解释为一个动作。这个例子也说明“轨迹”不一定很长：当答案已在首屏可见时，一轮 `STOP` 就是完整成功轨迹。

#### 输出与筛选

轨迹 judge 接收：

```text
任务描述
+ 顺序截图关键帧
+ Action 1 ... Action T
```

并输出：

```text
Correctness
Redundant
Optimized
First_Error_Step
Error_Type
Correct_Action
```

- `Correctness=True`：整条轨迹进入 `success/`；
- 失败且能定位首错：错误前进入 `positive/`，错误后进入 `negative/`；
- 无法形成有效正段：不保存。

CoMEM 训练时不是直接预测 judge 标签，而是从成功/正向轨迹中检索经验，压缩成连续 memory，再监督预测下一步 GUI action。

---

### 4.4.2 MMInA：多跳在线 Web GUI 任务

#### 真实任务输入

以下是仓库 `mmina/README.md` 给出的真实 task 17。原始 `intent_template` 与 `intent` 内容相同，下面完整展开一次，避免重复：

```json
{
  "sites": ["shopping"],
  "task_id": 17,
  "require_login": true,
  "storage_state": "./.auth/shopping_state.json",
  "start_url": "https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing",
  "geolocation": null,
  "intent": "For actions 'book a hotel','book a car', 'book a flight','search on the Youtube', 'search on the twitter', 'search some events', 'Find food', 'Travel Guide', 'Exchange dollars': the action is finished just after click the search button! Attention: If you think all the actions had been done, return the final url as the answer!!!\n\nHere are some reference urls:\nWiki: https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing\nRent a car: https://sg.trip.com/carhire/?channelid=14409&locale=en-SG&curr=USD\nBook a flight: https://www.momondo.com/\nBook a hotel: https://sg.trip.com/hotels/?locale=en-SG&curr=USD\nShopping: http://localhost:7770/\nSearch an event: https://www.eventbrite.com/\nTwitter: https://twitter.com/home\nYoutube: https://www.youtube.com/\nFind food: https://www.timeout.com/\nExchange dollars: https://www.xe.com/\nTravel Guide: https://www.nomadicmatt.com\n\nQuestion: Which city has a red tower, Tokyo or San Francisco? Help me check some events there.",
  "require_reset": false,
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {
      "must_include": ["eventbrite", "tokyo"]
    },
    "reference_url": "",
    "program_html": [],
    "string_note": "",
    "reference_answer_raw_annotation": []
  },
  "intent_template_id": 348,
  "cnt_hop": 3,
  "procedure": ["kiwix", "event", "end"],
  "shop": "",
  "city": "tokyo",
  "flight": "tyo"
}
```

#### 这个输入要求完成什么

任务包含两个语义子目标：

1. 在 Wikipedia/Kiwix 判断 Tokyo 与 San Francisco 中哪个城市有 red tower；
2. 根据答案 Tokyo，前往 Eventbrite 搜索当地活动。

`procedure=["kiwix","event","end"]` 只给出 hop-level 流程，不给鼠标级 ground-truth actions。

#### 数据集中的标准输出

MMInA 没有规定唯一自然语言答案，判分条件是：

```json
{
  "must_include": ["eventbrite", "tokyo"]
}
```

当前 evaluator 会把 Agent 最终 `STOP.answer` 与当前页面 URL 拼接后小写化，检查是否同时包含：

```text
eventbrite
tokyo
```

所以一种可通过的输出可能是：

```json
{
  "name": "stop",
  "arguments": {
    "answer": "Tokyo. Event results are available on https://www.eventbrite.com/d/japan--tokyo/events/",
    "reasoning": "The red tower is in Tokyo and the Eventbrite search has been completed."
  }
}
```

这不是唯一正确措辞；只要任务状态和 evaluator 条件满足即可。

#### 内部动作

MMInA JSON 不保存专家动作序列。下面是当前 CoMEM Agent 可能形成的解释性在线轨迹：

```text
Step 1  TYPE("Tokyo red tower", field="Wikipedia search")
Step 2  CLICK("search result about Tokyo Tower")
Step 3  CONTENT_ANALYZER("确认 Tokyo Tower 为红白色塔")
Step 4  GOTO_URL("https://www.eventbrite.com/")
Step 5  TYPE("Tokyo events", field="event search")
Step 6  CLICK("search")
Step 7  STOP("Tokyo; <Eventbrite Tokyo result URL>")
```

每一步真实输入都是：

```text
当前 screenshot
+ intent
+ 最近动作历史
+ 剩余动作数
+ 可选 memory
```

动作执行后 Playwright 获取下一张截图，因此它是在线多轮 GUI 交互，不是静态 QA。

---

### 4.4.3 Multimodal-Mind2Web：带截图、HTML 和专家动作的 step-level 数据

#### 真实样本

以下记录来自官方 `osunlp/Multimodal-Mind2Web` 的 `test_website` split，task/annotation ID 为：

```text
annotation_id = 013781df-4391-4533-bcb1-15f6819064f6
website       = tiktok.music
domain        = Entertainment
subdomain     = Music
```

完整任务输入：

```text
What are the romantic reggae musics from BCD Studio
that can be used in tik tok series in andorra
```

其人类演示动作序列 `action_reprs` 为：

```json
[
  "[label]   -> CLICK",
  "[div]  Andorra -> CLICK",
  "[span]  TikTok Series -> CLICK",
  "[span]  Reggae -> CLICK",
  "[span]  Romantic -> CLICK",
  "[input]   -> TYPE: BCD Studio",
  "[button]  Search -> CLICK"
]
```

语义展开：

```text
Step 0  点击地区选择框
Step 1  点击 Andorra
Step 2  点击 TikTok Series 使用场景
Step 3  点击 Reggae 风格
Step 4  点击 Romantic 情绪
Step 5  在搜索框输入 BCD Studio
Step 6  点击 Search
```

#### 单个 step 的完整输入/输出

Multimodal-Mind2Web 被展开为 action-level rows。该任务第 0 步的真实记录包括：

```json
{
  "action_uid": "79c4a963-4aa9-49c1-9257-6b0d5069c551",
  "annotation_id": "013781df-4391-4533-bcb1-15f6819064f6",
  "confirmed_task": "What are the romantic reggae musics from BCD Studio that can be used in tik tok series in andorra",
  "operation": {
    "original_op": "CLICK",
    "value": "",
    "op": "CLICK"
  },
  "target_action_index": "0",
  "target_action_reprs": "[label]   -> CLICK",
  "pos_candidates": [
    {
      "tag": "label",
      "backend_node_id": "110",
      "is_original_target": true,
      "is_top_level_target": true,
      "bounding_box_rect": "356,461,320,34",
      "is_clickable": true
    },
    {
      "tag": "input",
      "backend_node_id": "828",
      "is_original_target": false,
      "is_top_level_target": false,
      "placeholder": "Please select",
      "value": "United States",
      "bounding_box_rect": "369,467,272,22"
    }
  ]
}
```

实际完整 row 还包含：

- 动作前网页 `screenshot`；
- `raw_html`；
- `cleaned_html`；
- 大量 `neg_candidates`。

HTML 和图像体积很大，上面的代码块保留了完整任务语义、正目标和监督输出，没有把无关的数百个负候选逐项复制。

#### 数据集的输出是什么

官方离线设定不是生成一句最终答案，而是逐步预测：

```text
目标元素 + 操作类型 + 可选输入值
```

例如第 5 步输出：

```json
{
  "target": "[input]",
  "operation": "TYPE",
  "value": "BCD Studio"
}
```

官方可计算 Element Accuracy、Operation F1、Step Success Rate。

但 CoMEM 仓库对 Mind2Web 采用了另一种在线改造：浏览器执行整条任务，最后由 VLM 根据最后 5 张截图与 `STOP.answer` 判断 `SUCCESS/NOT SUCCESS`。因此论文中的 Mind2Web Task Accuracy 不能直接等同于官方离线 step accuracy。

---

### 4.4.4 WebVoyager：只有任务目标、没有固定动作轨迹的动态网站评测

#### 真实完整输入

官方 `WebVoyager_data.jsonl` 的真实样本 `Allrecipes--0`：

```json
{
  "web_name": "Allrecipes",
  "id": "Allrecipes--0",
  "ques": "Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people.",
  "web": "https://www.allrecipes.com/"
}
```

官方 `reference_answer.json` 在 `Allrecipes.answers` 下给出的对应参考为：

```json
{
  "id": 0,
  "type": "possible",
  "ans": "'Vegetarian Four Cheese Lasagna', 4.6-star, 181 reviews, Servings 8"
}
```

`type="possible"` 表示这是可接受参考答案，而不是唯一 golden answer。任务要求 suitable for 6 people，但参考答案写的是 `Servings 8`，这是原始标注内部的不完全一致，汇报时应保留而不是改写。

任务与参考答案文件不附带：

- 专家 click 坐标；
- 标准动作序列；
- 静态页面快照。

原因是网页内容、搜索结果和页面布局会变化。

#### 解释性内部动作

一个可能的在线 rollout：

```text
Step 1  TYPE("vegetarian lasagna", field="Allrecipes search")
Step 2  CLICK("search")
Step 3  浏览搜索结果，检查 rating 与 review count
Step 4  CLICK("Vegetarian Four Cheese Lasagna")
Step 5  检查评分、评论数和 servings
Step 6  若需要，换算或调整至 6 人份
Step 7  STOP("<recipe name、rating、reviews、servings/adjustment>")
```

这里没有唯一动作路径。Agent 可以先筛品牌、先筛评分或直接打开候选，只要最终页面和答案证明任务完成即可。

#### CoMEM 仓库中的输出与判分

Agent 最终输出形如：

```json
{
  "name": "stop",
  "arguments": {
    "answer": "Vegetarian Four Cheese Lasagna; rating 4.6; 181 reviews; recipe serves 8, with quantities adjusted to serve 6.",
    "reasoning": "The recipe satisfies the vegetarian, rating and review-count constraints, with serving information reported."
  }
}
```

然后 evaluator 输入：

```text
TASK: Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people.
Result Response: <STOP.answer>
+ 最后 5 张 trajectory screenshots
```

VLM judge 输出：

```text
<result>SUCCESS</result>
```

或：

```text
<result>NOT SUCCESS</result>
```

所以 WebVoyager 的 ground truth 更接近“由截图证明任务是否完成”，而不是一个固定字符串答案。论文还使用了 WebSight achievable subset，但精确 allowlist 没有随当前仓库发布。

---

### 4.4.5 GUI-Odyssey：真实移动端截图—动作演示轨迹

#### 真实完整 episode

以下是真实公开 episode：

```json
{
  "episode_id": "5342205429089853",
  "h": 2992,
  "w": 1344,
  "device_name": "Pixel 8 Pro",
  "category": "General_Tool",
  "app": ["Todoist", "Setting"],
  "meta_task": "Turn on/off notifications for any app on the phone and open the app.",
  "task": "Turn on/off notifications for any app on the phone and open the app.",
  "instruction": "Enable or disable notifications for any application on your device, and then launch the Todoist app via the Settings app.",
  "step_length": 11
}
```

#### 真实完整动作序列

坐标按照 GUI-Odyssey 规范归一化到 `[0,1000]`。每一步同时关联对应截图：

```json
[
  {
    "step": 0,
    "screenshot": "5342205429089853_0.png",
    "action": "CLICK",
    "info": [[882, 837], [882, 837]],
    "ps": "[(882, 837)]"
  },
  {
    "step": 1,
    "screenshot": "5342205429089853_1.png",
    "action": "CLICK",
    "info": [[433, 490], [433, 490]],
    "ps": "[(433, 490)]"
  },
  {
    "step": 2,
    "screenshot": "5342205429089853_2.png",
    "action": "CLICK",
    "info": [[215, 624], [215, 624]],
    "ps": "[(215, 624)]"
  },
  {
    "step": 3,
    "screenshot": "5342205429089853_3.png",
    "action": "CLICK",
    "info": [[813, 78], [813, 78]],
    "ps": "[(813, 78)]"
  },
  {
    "step": 4,
    "screenshot": "5342205429089853_4.png",
    "action": "TEXT",
    "info": "Todoist",
    "ps": ""
  },
  {
    "step": 5,
    "screenshot": "5342205429089853_5.png",
    "action": "CLICK",
    "info": [[380, 139], [380, 139]],
    "ps": "[(380, 139)]"
  },
  {
    "step": 6,
    "screenshot": "5342205429089853_6.png",
    "action": "CLICK",
    "info": [[286, 503], [286, 503]],
    "ps": "[(286, 503)]"
  },
  {
    "step": 7,
    "screenshot": "5342205429089853_7.png",
    "action": "CLICK",
    "info": [[862, 407], [862, 407]],
    "ps": "[(862, 407)]"
  },
  {
    "step": 8,
    "screenshot": "5342205429089853_8.png",
    "action": "CLICK",
    "info": "KEY_HOME",
    "ps": ""
  },
  {
    "step": 9,
    "screenshot": "5342205429089853_9.png",
    "action": "CLICK",
    "info": [[636, 531], [636, 531]],
    "ps": "[(636, 531)]"
  },
  {
    "step": 10,
    "screenshot": "5342205429089853_10.png",
    "action": "COMPLETE",
    "info": "",
    "ps": ""
  }
]
```

这里 `TEXT("Todoist")` 是明确标注的输入动作，`KEY_HOME` 表示返回 Android Home，最后 `COMPLETE` 表示演示结束。

仅从坐标不能可靠恢复每个按钮的语义名称，必须结合对应 screenshot 查看；因此不能在没有图像的情况下武断地把每个坐标写成某个按钮。

#### 输入与输出

对第 \(t\) 步：

```text
输入 = instruction + 当前 screenshot X_t + 历史动作
输出 = action A_t + info
```

动作空间包括：

```text
CLICK, SCROLL, LONG_PRESS, TEXT/TYPE,
COMPLETE, IMPOSSIBLE, HOME, BACK, RECENT
```

论文 OOD 表报告 AMS：

- High Level：高层动作/流程是否匹配；
- Low Level：具体动作类型、坐标或文本是否匹配。

官方 evaluator 的关键规则是：

- 预测动作类型首先必须与 ground truth 一致；
- `CLICK/LONG_PRESS`：预测点落入 `sam2_bbox`，或者与参考点的归一化欧氏距离不超过 `0.14`；
- `SCROLL`：比较滑动方向；
- `TYPE`：一方文本包含另一方可直接通过，否则要求 normalized Levenshtein similarity 不低于 `0.5`；
- `COMPLETE/IMPOSSIBLE/HOME/BACK/RECENT`：动作类型一致即可。

原始 annotation 与 evaluator 命令还存在一层格式转换：

```text
原始 TEXT              → evaluator TYPE
CLICK + KEY_HOME       → PRESS_HOME
CLICK + KEY_BACK       → PRESS_BACK
CLICK + KEY_APPSELECT  → PRESS_RECENT
原始 INCOMPLETE        → evaluator IMPOSSIBLE
```

因此读原始 JSON 时不能把 `CLICK(info="KEY_HOME")` 错当成普通坐标点击。

与 MMInA/WebVoyager 不同，GUI-Odyssey 确实提供人类专家的完整动作轨迹，因此可以逐步监督和逐步匹配。

---

### 4.4.6 OSWorld：桌面虚拟机任务与环境状态 evaluator

#### 真实完整任务输入

以下是 OSWorld 官方文档中的 Chrome PDF 任务：

```json
{
  "id": "e1e75309-3ddb-4d09-92ec-de869c928143",
  "instruction": "Computer, can you turn the webpage I'm looking at into a PDF file, save it to my Desktop with the default filename and set the margins to none?",
  "source": "https://in5stepstutorials.com/google-chrome/save-web-page-as-pdf-in-chrome.php",
  "config": [
    {
      "type": "launch",
      "parameters": {
        "command": [
          "google-chrome",
          "--remote-debugging-port=1337"
        ]
      }
    },
    {
      "type": "launch",
      "parameters": {
        "command": [
          "socat",
          "tcp-listen:9222,fork",
          "tcp:localhost:1337"
        ]
      }
    },
    {
      "type": "chrome_open_tabs",
      "parameters": {
        "urls_to_open": [
          "https://lilianweng.github.io/posts/2023-06-23-agent/"
        ]
      }
    }
  ],
  "related_apps": ["chrome"],
  "evaluator": {
    "func": "compare_pdfs",
    "result": {
      "type": "vm_file",
      "path": "/home/user/Desktop/LLM Powered Autonomous Agents _ Lil'Log.pdf",
      "dest": "LLM Powered Autonomous Agents _ Lil'Log.pdf"
    },
    "expected": {
      "type": "pdf_from_url",
      "path": "https://lilianweng.github.io/posts/2023-06-23-agent/",
      "dest": "LLM Powered Autonomous Agents _ Lil'Log_gold.pdf"
    }
  },
  "proxy": true,
  "fixed_ip": false,
  "possibility_of_env_change": "medium"
}
```

`config` 不是 Agent 的动作答案，而是 benchmark harness 在任务开始前执行的环境初始化：

1. 启动 Chrome；
2. 建立调试端口转发；
3. 打开指定网页。

#### Agent 的输入

正式执行时，每一步输入是：

```text
instruction
+ 当前 Ubuntu VM screenshot
+ 可选 accessibility tree
+ 历史动作
```

#### 内部动作

OSWorld 任务配置不要求提供唯一专家轨迹。一个合理的解释性执行路径是：

```text
Step 1  按 Ctrl+P 打开 Chrome Print
Step 2  选择 Destination = Save to PDF
Step 3  展开 More settings
Step 4  设置 Margins = None
Step 5  点击 Save
Step 6  在文件选择器中保持默认文件名并选择 Desktop
Step 7  确认保存
Step 8  STOP
```

不同 Agent 可以通过菜单点击或快捷键完成，动作序列不必与上面相同。

OSWorld 官方支持的动作后端不止一种，包括 `pyautogui`、`computer_13`、Claude/Gemini computer-use 风格。常见结构化动作可覆盖：

```text
MOVE_TO, CLICK, RIGHT_CLICK, DOUBLE_CLICK, DRAG_TO,
SCROLL, TYPING, PRESS, HOTKEY, WAIT, FAIL, DONE
```

在 `pyautogui` 空间中，模型也可以直接输出：

```python
pyautogui.click(500, 300)
pyautogui.hotkey("ctrl", "p")
pyautogui.write("filename.pdf")
```

这些是 Agent 某次运行产生的动作，不是数据集预先规定的唯一正确序列。

#### 真正的输出

这个任务的输出不是一句：

```text
"saved successfully"
```

而是 VM 文件：

```text
/home/user/Desktop/LLM Powered Autonomous Agents _ Lil'Log.pdf
```

evaluator 执行：

```text
compare_pdfs(
  result=<Agent 生成的 PDF>,
  expected=<由原网页生成的 gold PDF>
)
```

只有文件存在且内容满足比较规则才算成功。Agent 即使声称“已完成”，但没有生成正确 PDF，Success Rate 仍为 0。

一次实际 OSWorld rollout 通常保存为：

```text
initial_state.png
step_1_<timestamp>.png
step_2_<timestamp>.png
...
traj.jsonl
recording.mp4
result.txt
```

其中 `traj.jsonl` 的 step 记录包含：

```json
{
  "step_num": 1,
  "action": "pyautogui.<本轮动作>",
  "response": "<模型原始输出>",
  "reward": 0,
  "done": false,
  "info": {},
  "screenshot_file": "step_1_<timestamp>.png"
}
```

它是待评模型运行后生成的日志，而不是公开的 ground-truth action list。

当前 CoMEM 仓库没有 OSWorld environment adapter、desktop action executor 或 evaluator 接口，论文中的 OSWorld OOD 结果不能直接由当前 `run.py` 重跑。

---

### 4.4.7 六类数据的输入/输出差异总结

| 数据集 | 静态输入 | 是否含截图 | 是否含专家动作 | 监督/输出 | CoMEM 论文中的用途 |
|---|---|---:|---:|---|---|
| CoMEM trajectories | 自动生成 task + URL | 是，多轮 | 是，Agent rollout | 轨迹成功/失败、首错、下一动作 | 建 memory、训练编码器 |
| MMInA | task JSON + URL + eval 规则 | 否，运行时截图 | 否 | 最终答案/URL/HTML 条件 | 主在线评测 |
| Multimodal-Mind2Web | task + HTML + step screenshot | 是 | 是，人类演示 | target element + CLICK/TYPE/SELECT | 主评测的来源数据 |
| WebVoyager | website + question + start URL | 否，运行时截图 | 否 | VLM 判断 SUCCESS/NOT SUCCESS | 主在线评测 |
| GUI-Odyssey | instruction + mobile screenshots | 是 | 是，人类演示 | action sequence / AMS | mobile OOD |
| OSWorld | instruction + VM setup + evaluator | 运行时 screenshot | 通常不要求唯一轨迹 | 最终 VM 状态/文件 / SR | desktop OOD |

最关键的区别是：

\[
\text{Mind2Web/GUI-Odyssey}
\approx
\text{离线专家轨迹数据}
\]

而：

\[
\text{MMInA/WebVoyager/OSWorld}
\approx
\text{任务配置 + 在线环境 + 结果 evaluator}
\]

CoMEM trajectory 则是作者让 Agent 在在线环境中 rollout 后自行收集出来的经验库。

## 4.5 Baseline 分组

1. **闭源基础模型**：GPT-4o、Gemini-Pro-Vision、Claude-4。仅作参考，不参与表中最佳/次佳排名。
2. **开源基础模型**：Qwen2-VL-7B、Qwen2.5-VL-7B、GLM-4.1V-9B、Qwen2.5-VL-32B。
3. **GUI 专项微调**：UI-TARS-1.5、CogAgent、WebSight。
4. **记忆增强**：
   - UI-TARS-1.5-7B + Text-based Memory；
   - UI-TARS-1.5-7B + CoMEM；
   - Qwen2.5-VL-7B + Text-based Memory；
   - Qwen2.5-VL-7B + CoMEM。

论文称主表结果均由作者在统一环境、任务采样和评价协议下复现，这一点优于直接抄不同论文数字；但闭源模型版本、调用日期、judge 版本和随机重复次数未完整披露。

## 4.6 训练与推理成本

论文明确报告：

- 数据飞轮：约 \$4,000，100k+ 轨迹；
- 训练样本：1,500 条；
- 训练参数：Q-Former + LoRA，约 1.2%；
- LoRA rank：16；
- 训练硬件：单张 NVIDIA H100；
- 训练时长：20 小时；
- 每条轨迹：8 个连续向量；
- 训练实例：轨迹逐步切分，每步配 top-3 记忆。

仓库训练脚本补充：

- Qwen2.5-VL-7B-Instruct；
- bf16、FlashAttention-2、gradient checkpointing、DeepSpeed ZeRO-3；
- 1 epoch；
- global batch size 32；
- learning rate \(5\times10^{-5}\)；
- weight decay 0.1；
- warmup ratio 0.03；
- cosine scheduler；
- LoRA alpha 32、dropout 0.05；
- 图像像素范围 \(256\times28^2\) 到 \(1280\times28^2\)。

但当前脚本设置 `CUDA_VISIBLE_DEVICES=4,5,6,7`、`NUM_DEVICES=4`，与论文“单 H100 20 小时”冲突。训练脚本还缺少实际 `data_path` 和 `output_dir`。因此现仓库不能直接作为论文成本声明的逐命令复现证据。

论文没有报告：

- 100k 轨迹采集所用 GPU/API 调用量分解；
- H100 具体显存规格；
- FAISS 建库耗时、索引大小和峰值内存；
- 完整训练峰值显存；
- scaling 实验的重复次数、误差条或置信区间；
- 每个模型端到端 token/FLOPs 成本。

---

# 5. 做了哪些实验、每项指标是什么、怎样支撑结论

## 5.1 指标解释

### Task Accuracy / Acc.

\[
\mathrm{TaskAcc}
=\frac{\#\text{被判定成功的任务}}{\#\text{有效评测任务}}\times100\%.
\]

MMInA 将模型最终答案和 ground truth 交给 LLM 判断；Mind2Web 和 WebVoyager 将任务描述与轨迹截图序列交给 VLM 判断是否完成。

它是整条任务级的 0/1 指标，不是单步动作准确率。优点是直接反映最终完成；缺点是受 judge 偏差、网页变化及任务过滤影响。

### Success Rate（SR）

OSWorld 中：

\[
\mathrm{SR}
=\frac{\#\text{成功完成任务}}{\#\text{总任务}}\times100\%.
\]

与 Task Accuracy 概念接近，但遵循 OSWorld 原始执行式评测协议。

### Action Matching Score（AMS）

GUI-Odyssey 中衡量预测动作序列与参考动作的匹配程度。High Level 更关注高层操作语义/流程，Low Level 更关注具体执行动作。论文没有在正文重新给出 AMS 的精确定义和匹配公式，应以 GUI-Odyssey 官方 evaluator 为准。

### Time

一条完整轨迹平均耗时，单位为分钟/trajectory。它同时受模型推理、网页加载、动作数、重试和网络状态影响，不是纯模型 latency。

### Avg.

主表最后一列的总体平均。正文没有明确写出它是七列宏平均、按任务数加权平均还是其他聚合。由表中数字看，它并非简单七列算术平均。因此只能把它解释为作者的总体聚合分数，不能在缺少每列样本数时自行重算。

## 5.2 实验一：主结果——连续记忆是否有效

### Qwen2.5-VL-7B

- 无记忆：Wiki 36.7，MMInA Shop 15.5，WebVoyager 40.0，Avg 14.4。
- 文本记忆：Wiki 34.2，MMInA Shop 31.4，WebVoyager 44.0，Avg 22.2。
- CoMEM：Wiki 47.4，MMInA Shop 45.0，WebVoyager 54.5，Avg 31.7。

相对无记忆，CoMEM 的表中 Avg. 增加：

\[
31.7-14.4=17.3\ \text{个百分点}.
\]

相对文本记忆增加：

\[
31.7-22.2=9.5\ \text{个百分点}.
\]

这支撑两个层次的观点：

1. 外部经验本身有效：文本记忆总体优于无记忆；
2. 连续多模态表示比文本记忆更有效：CoMEM 在绝大多数列上进一步提升。

但 Mind2Web Service 是反例：CoMEM 17.7%，文本记忆 16.6%，而 GLM-4.1V-9B 达到 33.3%。因此“所有领域都达到最好”不成立。

### UI-TARS-1.5-7B

- 专项模型行 Avg.：13.2；
- + Text Memory：10.0；
- + CoMEM：23.8。

CoMEM 对另一个 backbone 也有效，支撑一定的模型可迁移性；文本记忆反而降低总体表现，支撑“离散提示会引入噪声”的论点。

正文称 UI-TARS 从“6.6% 到 23.8%”，但表中 UI-TARS baseline Avg. 是 13.2%。这是论文内部不一致，汇报应以表格为主并提示作者核验。

### 与闭源模型比较

- CoMEM Avg. 31.7；
- GPT-4o 27.8；
- Gemini-Pro-Vision 30.4；
- Claude-4 28.8。

因此“7B 开源模型达到可比闭源水平”由总体结果支持。WebVoyager 上 CoMEM 54.5，超过闭源三者的 31.8/47.7/40.9。

但闭源模型被排除出排名，而且具体 API snapshot 未给出；“超过 SOTA”更适合表述为“在作者统一复现设置下超过所列闭源模型”。

## 5.3 实验二：记忆库规模 scaling

作者在 MMInA Shopping 上改变 memory bank size \(M\)，并对固定检索条数拟合：

\[
\mathrm{Acc}(M)=a+b\log M.
\]

固定检索条数取 \(\{3,10,50,100\}\)，通过普通最小二乘分别估计 \(a,b\)。观察：

- bank size 增大，准确率总体上升；
- 检索条数越大，曲线提升越陡。

支撑观点：大记忆库并非只增加冗余，覆盖更多任务/界面能改善检索增强决策。

严谨性限制：

- 正文公式写 `Acc(m)`，同时又用 \(m\) 表示检索样本数，符号与 memory size \(M\) 混用；
- 没有给出 \(a,b,R^2\)、显著性、误差条和随机种子；
- 只在一个任务域上拟合；
- 因此是 empirical scaling trend，而非可外推的定律。

## 5.4 实验三：检索深度 scaling

作者改变 top-\(K\) 检索数，在 \(\log K\) 上用三次多项式拟合 MMInA Shopping 准确率。

结果：

- CoMEM 随 \(K\) 增大保持上升趋势；
- Text-MEM 在约 \(K=10\) 后下降。

这直接支撑论文最核心的“连续记忆可扩展、文本上下文不可无限扩展”论点。

但三次多项式只是曲线拟合工具，没有机制保证单调；原始重复试验和方差未报告。图中趋势不能证明无限扩大 \(K\) 仍会提升。

## 5.5 实验四：跨 GUI 域泛化

训练和记忆来源都是 web，直接测试 mobile GUI-Odyssey 与 desktop OSWorld。

### GUI-Odyssey（AMS）

- Baseline：High 22.38，Low 45.58；
- Text Memory：High 24.42，Low 37.35；
- CoMEM：High 27.41，Low 44.90。

CoMEM 提升高层匹配，但 Low Level 比 baseline 低 0.68 个百分点。说明连续记忆更可能迁移高层程序知识，未证明低层动作定位全面提升。

### OSWorld（SR）

- Baseline Overall：26.40；
- Text Memory：24.70；
- CoMEM：26.73。

CoMEM 总体只提升 0.33 个百分点，幅度很小；Daily 从 25.60 升至 28.21，Office 和 Professional 小幅上升。文本记忆各列普遍下降。

结论应表述为：CoMEM 在跨平台 OOD 下避免了文本记忆的明显负迁移，并在若干高层/日常任务上改善；“强泛化”证据是正向但幅度有限。

## 5.6 实验五：推理时间

MMInA 两类任务结果：

- Wikipedia：baseline 2.33 min / 72.4 Acc；Text 1.50 / 77.1；CoMEM 1.58 / 81.3。
- Shopping：baseline 1.57 min / 68.9 Acc；Text 2.12 / 73.5；CoMEM 2.13 / 76.8。

解释：

- Wikipedia 中记忆帮助 Agent 更快找到路径，轨迹更短，总时长下降；
- Shopping 中记忆增加约 0.56 分钟，但准确率增加 7.9 个百分点；
- CoMEM 与文本记忆的端到端时间几乎相同，而准确率更高。

这支持“没有显著端到端额外延迟”的弱结论，但不支持严格的“无额外推理开销”：Shopping 时间增加约 36%，且论文没有分离检索、压缩、模型 prefill、网页交互时间。

还要注意本表准确率 72.4/68.9 与主表相同领域的 36.7/15.5 不一致，显然使用了不同任务子集或设置，但论文未清楚说明。因此不能把两张表的准确率直接横向比较。

## 5.7 实验六：训练数据规模

- 500 条：Wiki 39.30，Shop 33.30；
- 1,000 条：43.83，39.00；
- 1,500 条：47.40，45.00；
- 2,000 条：45.00，42.60。

支撑：

- 从 500 到 1,500 条持续提升；
- 仅 1,500 条即可达到主结果，显示参数/样本效率。

2,000 条下降不必然说明“已经对齐完成”。也可能来自数据质量下降、采样差异、优化步数变化或随机波动。论文没有误差条和多随机种子，不能据此证明 1,500 是普适最优规模。

## 5.8 实验七：定性案例

### 购物案例

无记忆模型选择超预算商品；CoMEM 模型比较候选并验证颜色数、用途和价格，说明记忆有助于长程约束保持。

### 信息检索案例

问题：“上海市中心是否有一座带多个球体的电视塔？”

- 无记忆：搜索过宽的 “Shanghai”，反复滚动无关页面；
- CoMEM：搜索 “Shanghai TV tower”，找到东方明珠并用视觉和文本证据核验。

案例说明记忆可能改善查询规划和探索效率，但两个成功案例不足以排除 cherry-picking；最好补充失败类型统计和检索到的具体轨迹可视化。

## 5.9 论文还缺少哪些关键消融

论文没有完整报告：

1. 8 个 memory token 与 4/16/32 个 token 的消融；
2. 文本-only、图像-only、multimodal retrieval key 的拆分；
3. CLIP/FAISS 检索质量指标（Recall@K、MRR、oracle retrieval）；
4. 随机记忆、最不相似记忆、无检索但同长度 soft token 控制；
5. Q-Former 深度、共享层参数与非共享层消融；
6. 只训练 Q-Former、只训练 LoRA、二者同时训练的拆分；
7. 自动 judge 的人工一致率、误报和漏报；
8. 自动数据与公开数据的比例及各自贡献；
9. 多随机种子、方差和显著性检验；
10. 记忆污染、过期网页、重复轨迹和训练—测试泄漏检查。

因此论文已经证明“系统有效”，但对“为什么有效、哪一部分贡献最大”的因果归因仍不充分。

---

# 6. 复现需要做哪些准备

## 6.1 推荐分三级复现

### Level 1：只复现推理结果

目标：使用作者 checkpoint 和发布的 memory/index，在一个固定 benchmark 子集上比较 no-memory、Text-MEM、CoMEM。

这是成本最低、最适合先验证的方案。

### Level 2：复现 memory encoder 训练

目标：冻结 Qwen2.5-VL-7B，用 1,500 条轨迹训练 Q-Former + LoRA，再评测。

可以验证 1.2% 参数、样本效率和 20 小时训练声明。

### Level 3：复现完整数据飞轮

目标：从 Mind2Web seed 开始，经网站发现、任务生成、rollout、judge 得到新记忆，再训练和评测。

该层受真实网站漂移、API 成本、合规和 judge 偏差影响最大，最难做到逐数字复现。

## 6.2 数据准备

需要：

1. Mind2Web train 任务作为 \(\mathcal Q_0\)；
2. MMInA、Multimodal-Mind2Web、WebVoyager 测试任务；
3. OOD 时再准备 GUI-Odyssey、OSWorld；
4. 作者 Hugging Face 轨迹数据；
5. 每条轨迹至少包含：
   - `task_description`；
   - URL / environment / domain；
   - 每步截图（仓库使用 base64）；
   - 模型 response；
   - 可解析 action；
   - `total_rounds`；
   - success 标志；
6. 论文所用的任务 allowlist、跳过列表和网站 snapshot。

必须做数据审计：

- 训练任务与测试任务文本/URL/截图近重复检查；
- 同一网站模板泄漏检查；
- 无法访问、验证码、登录和地区限制统计；
- 轨迹成功标签人工抽检；
- PII、版权和敏感页面过滤。

## 6.3 模型与 checkpoint

核心模型：

- Actor / rollout：Qwen2.5-VL-32B-Instruct；
- 主实验 backbone：Qwen2.5-VL-7B-Instruct；
- 另一 backbone：UI-TARS-1.5-7B；
- Judge：SEAgent-1.0-7B；
- 检索：`openai/clip-vit-base-patch32`；
- continuous compressor：8-query Q-Former；
- grounding fallback：UI-TARS-1.5-7B。

作者发布的 checkpoint：

- `WenyiWU0111/lora_qformer_test_V4-700_merged`；
- `WenyiWU0111/lora_qformer_uitars_test_V1-400_merged`。

闭源 baseline 还需固定 GPT-4o、Claude、Gemini 的准确模型版本和调用日期，否则 API 漂移会改变结果。

## 6.4 硬件准备

论文训练口径：

- 1×NVIDIA H100；
- 20 小时。

完整复现还需要：

- 7B CoMEM 推理 GPU，建议至少 80GB 显存以减少自定义多模型结构的 OOM 风险；
- 32B rollout 可用多卡 H100/A100 或高显存推理服务；
- 足够本地存储保存 100k+ 多截图轨迹；
- CPU 内存和磁盘用于 CLIP 编码及 FAISS 建库；
- Playwright 浏览器执行节点；
- 若并行采集，应隔离浏览器 profile、缓存和账号状态。

论文没有给出最低显存；上述是工程建议，不是论文报告。

## 6.5 软件与服务

根据代码导入至少需要：

- Python 3.10+；
- PyTorch、Transformers、PEFT；
- DeepSpeed、FlashAttention-2、Liger Kernel；
- qwen-vl-utils；
- FAISS；
- Playwright；
- OpenAI Python client / vLLM OpenAI-compatible server；
- Pillow、NumPy、ujson、tqdm；
- streaming / MosaicML Streaming；
- SerpAPI key；
- Hugging Face 模型与数据访问。

当前仓库没有 `requirements.txt`、`pyproject.toml` 或 `environment.yml`，因此必须自行锁定依赖版本并记录 CUDA、driver、PyTorch 和 Transformers commit。

## 6.6 训练配置

按论文与脚本对齐：

```text
base model          Qwen/Qwen2.5-VL-7B-Instruct
memory queries      8 per trajectory
Q-Former depth      8
hidden size         3584
attention heads     16
head dim            224
shared layers       True
LoRA rank           16
LoRA alpha          32（脚本）
LoRA dropout        0.05（脚本）
training samples    1,500 trajectories
retrieved memories  top-3 per training instance
epochs              1（脚本）
global batch        32（脚本）
learning rate       5e-5（脚本）
weight decay        0.1（脚本）
warmup ratio        0.03（脚本）
scheduler           cosine（脚本）
precision           bf16
max steps/task      15（推理默认）
```

应额外固定：

- 1,500 条轨迹的精确 ID；
- 每条轨迹拆成多少 step instance；
- train shuffle seed；
- top-3 检索索引版本；
- 图像 resize/min-max pixels；
- optimizer、gradient accumulation 和 checkpoint step；
- 选择 “700” checkpoint 的准则。

## 6.7 推理和评测协议

每种方法必须保持相同：

- 任务清单和顺序；
- viewport 1280×720；
- 最大步数 15；
- SOM 生成方式；
- grounding fallback；
- temperature/top-p/max tokens；
- 浏览器和网络条件；
- judge prompt、judge model；
- 无效网站跳过规则；
- 至少 3 个随机种子或多次运行。

分别记录：

- Task Accuracy / SR / AMS；
- 轨迹步数；
- 端到端时间；
- 检索时间；
- Q-Former 压缩时间；
- VLM prefill/decode 时间；
- 输入 token / continuous token 数；
- 峰值显存；
- judge 与人工抽检一致率。

## 6.8 当前仓库的直接复现阻塞项

在正式跑实验前需要修复或澄清：

1. 根目录 README 要求 `pip install -r requirements.txt`，但仓库中没有该文件。
2. `run_with_continuous_memory.sh` 传 `--checkpoint_path`，但 `run_baseline.sh` 不解析也不转发该参数。
3. Python argument parser 也没有声明 `--checkpoint_path`。
4. continuous memory 示例脚本中的反斜杠与注释布局会中断 shell 命令。
5. 训练脚本的 `--output_dir ''` 为空，且未传 `--data_path`。
6. `data.py` 硬编码从 `training_data` 加载，并硬编码只保留部分 dataset 名。
7. 若 `similar_trajectories` 为空，代码尝试访问 `similar_trajectories[-1]`，可能报错。
8. 训练脚本配置 4 张 GPU，而论文报告单 H100。
9. 推理模型类中存在硬编码路径 `CoMEM-Agent/CoMEM-Agent-train` 和 localhost 端口。
10. `create_direct_vllm_model` 使用 `args.getattr(...)`，标准 `argparse.Namespace` 没有该方法。
11. 代码将连续记忆强制截断/补齐为 24 个向量，与论文的大 \(K\) scaling 设置对应关系不清。
12. README 的数据规模/成本与论文不同。
13. 论文主表与 latency 表同名任务准确率不一致，需得到作者的精确 split。

所以当前最稳妥路线是先使用发布 checkpoint，修通单任务 CoMEM 推理，再逐级扩大；不建议直接启动 100k 轨迹全流程。

## 6.9 建议的复现验收顺序

1. **单样本 smoke test**：同一任务下 no-memory 与 CoMEM 都能完整执行。
2. **表示检查**：每条轨迹输出恰好 \(8\times3584\)，top-3 合计 24 token。
3. **检索检查**：人工查看 top-3 是否与任务语义/视觉相关。
4. **小规模 overfit**：几十个训练实例上 loss 明显下降。
5. **论文案例复现**：购物与上海电视塔案例。
6. **固定子集 A/B test**：先跑 MMInA Wiki/Shopping 各 20 条。
7. **完整主实验**：固定任务清单与 judge。
8. **scaling 实验**：逐一改变 bank size 和 \(K\)，至少 3 次运行并报告均值/标准差。
9. **OOD 实验**：最后测试 GUI-Odyssey 与 OSWorld，避免环境配置问题混入主结论。

---

# 7. 对论文的总体评价

## 7.1 优点

1. **问题抓得准**：GUI 轨迹既长又高度视觉化，文本 memory 确实不是理想介质。
2. **系统组合合理**：检索解决选择，Q-Former 解决压缩，embedding injection 解决模型接入，数据飞轮解决规模。
3. **参数和数据效率高**：只用 1,500 条训练轨迹、约 1.2% 参数。
4. **实验覆盖较广**：web 主任务、mobile/desktop OOD、scaling、latency、训练规模与案例。
5. **跨 backbone 有证据**：Qwen2.5-VL 和 UI-TARS 都从 CoMEM 获益。

## 7.2 主要不足

1. “scaling law”证据偏经验性，缺少统计量和跨域验证。
2. 关键机制消融不足，不能分清收益来自视觉信息、soft prompt 容量、检索质量还是训练方式。
3. 自动 judge 同时用于数据筛选和评测，可能产生同源偏差与自强化。
4. 动态网站使 benchmark 难以精确复现，任务过滤可能带来选择偏差。
5. 数据飞轮存在隐私、版权、prompt injection、数据投毒和 popular-site bias。
6. 论文内部及论文—代码间存在多处数字/配置不一致。
7. 当前公开代码尚未达到“一键复现”状态。

## 7.3 最准确的结论边界

论文充分支持：

> 在作者的 GUI Agent 框架和评测设置内，将检索到的多模态轨迹压缩成连续前缀，比无记忆或文本记忆通常更有效；扩大记忆库和检索深度在已测试范围内呈正向趋势。

论文尚未充分证明：

> 连续记忆在任意模型、任意 GUI 域和任意规模下都会单调提升，或已形成可可靠外推的普适 scaling law。

---

# 8. 组会可直接使用的总结

本文的关键创新不是单独发明 Q-Former、FAISS 或自生成数据，而是把它们围绕 GUI 经验复用组织成闭环：

\[
\boxed{
\text{自动采经验}
\rightarrow
\text{检索相关经验}
\rightarrow
\text{压成 8 个连续向量}
\rightarrow
\text{注入冻结 VLM}
\rightarrow
\text{改善下一动作}
}
\]

它把 GUI Agent 的“学习新技能”从昂贵的全参再训练，部分转化为可扩展的非参数记忆增长。最值得后续研究的方向不是简单继续堆记忆，而是：

1. 不确定性感知的自适应检索 \(K\)；
2. 检索信用分配：哪些记忆真正帮助了哪个动作；
3. 分层/分片索引与记忆去重、过期和来源治理；
4. 对自动 judge 的校准与人类一致性评估；
5. 安全、隐私和 prompt-injection-aware 的记忆写入与读取；
6. 用严格消融确认视觉连续记忆的真实因果贡献。

---

## 参考定位

- 任务定义与记忆策略：`docs/tex/main.tex` 第 275–301 行；
- 数据飞轮：第 316–417 行；
- 连续记忆、检索和微调：第 419–449 行；
- 实验设置与 baseline：第 474–510 行；
- 主结果与 scaling：第 512–580 行；
- OOD、latency、训练规模：第 583–685 行；
- 限制、伦理与复现声明：第 691–713 行；
- 案例与任务生成 prompt：第 723–947 行；
- Q-Former 实现：`CoMEM-Agent-train/src_agent/training/qformer.py`；
- 连续记忆注入：`CoMEM-Agent-train/src_agent/training/qwenVL_compressor.py`；
- CLIP + FAISS：`CoMEM-Agent-Inference/memory/experience_memory.py`；
- 训练脚本：`CoMEM-Agent-train/scripts/finetune_lora_vision_test.sh`。

