# Situation-Indexed, Evidence-Grounded Continuous Memory for GUI Agents

> 面向 GUI Agent 的情境索引、证据驱动连续记忆研究构想  
> 基础系统：CoMEM-Agent  
> 核心目标：解决“任务文本、界面和应用都不相似，但当前局面需要复用同类历史经验”的记忆检索问题。

---

# 1. 一句话概括

CoMEM 已经解决：

> 检索到历史轨迹以后，如何把长截图—动作序列压缩成少量连续向量并注入冻结 VLM。

本研究希望进一步解决：

> 当前究竟处于什么 GUI situation，以及应该检索哪段真正有帮助的历史轨迹。

二者可以组合为：

```text
Current GUI State
    ↓
Situation Extractor
    ↓
Situation Hypergraph Retriever
    ↓
Retrieved Raw Evidence
    ↓
CoMEM Q-Former Compressor
    ↓
Continuous Memory Tokens
    ↓
Frozen GUI Agent
```

即：

\[
\boxed{\text{Indexing by Situation, Injecting by Evidence}}
\]

---

# 2. 研究动机

## 2.1 CoMEM 当前如何检索

CoMEM 将当前任务文本和首张截图编码为检索键：

\[
k_q=
\left[
\operatorname{CLIPText}(q);
\operatorname{CLIPImage}(I_1)
\right].
\]

历史轨迹也被编码到同一空间，经过 L2 归一化后，使用 FAISS inner product 检索 top-\(K\)：

\[
\mathcal R_K(q)=
\operatorname{TopK}_{\tau_i}
\operatorname{sim}(k_q,k_i).
\]

这种方式主要回答：

> 当前任务和哪条历史任务在文字、截图或整体语义上更相似？

## 2.2 仅靠相似度不够

GUI Agent 真正需要复用的往往不是相似页面，而是相同的：

- 信息交接关系；
- 当前任务阶段；
- 已完成与待完成子目标；
- 已满足与未满足约束；
- 页面或应用切换状态；
- 错误原因及恢复策略；
- 下一步需要消费的历史实体。

例如：

```text
历史任务：Waze 找到健身房 → Uber 输入该地点
当前任务：Google Maps 找到书店 → Lyft 输入该地点
```

两条轨迹的：

- App 不同；
- 任务文本不同；
- 实体不同；
- screenshot 不同。

但底层 situation 完全相同：

\[
\text{Source app acquires entity}
\rightarrow
\text{Target app consumes entity}.
\]

普通 similarity-first retrieval 可能无法召回；situation retrieval 则应把它们视为强相关经验。

---

# 3. 研究定位

不建议把研究定位成：

> 给 CoMEM 加一个超图。

更有价值的定位是：

> 在任务文本、App/网站与视觉界面都不相似时，根据共同的 GUI situation 找回真正有帮助的历史证据。

推荐题目：

> **Situation-Indexed, Evidence-Grounded Continuous Memory for GUI Agents**

超图是实现多变量联合依赖的一种数据结构，而不是研究贡献本身。

核心贡献应包括：

1. 将 retrieval index 与 injected evidence 解耦；
2. 使用 GUI situation 而非单纯 task/image similarity 建立记忆可达性；
3. 从成功与失败轨迹中抽取可迁移情境；
4. 构造 surface-disjoint associative retrieval evaluation；
5. 审计哪条 situation relation 召回了什么证据，以及该证据是否真正帮助任务完成。

---

# 4. 为什么 GUI 比长对话更适合 situation memory

对话中的 situation 往往是隐式的，例如：

- 用户价值观；
- 历史承诺；
- 话题之间的远距离因果关系。

GUI 中的 situation 可以从可观察状态中明确抽取：

```text
当前 App / 网站
当前页面
最近动作
动作结果
任务阶段
已完成子目标
待完成子目标
已知实体
未满足约束
页面切换
应用切换
成功/失败 outcome
```

GUI 还提供可验证信号：

- 页面是否发生变化；
- 新 App 是否打开；
- 输入框是否已填入；
- 筛选项是否选中；
- 文件是否生成；
- 子任务是否完成；
- evaluator 是否判定成功。

因此 situation 的定义、标注和验证都比一般对话更具体。

---

# 5. CoMEM 数据集中的对应案例

## 5.1 MMInA：跨网站实体交接

真实任务：

```text
判断 Tokyo 与 San Francisco 中哪个城市有 red tower，
然后在 Eventbrite 搜索该城市的活动。
```

依赖结构：

```text
Wikipedia
  ↓
acquired_entity = Tokyo
  ↓
Eventbrite
  ↓
pending_subgoal = search events in Tokyo
```

可抽取 situation：

```yaml
type: Handoff
source_site: Wikipedia
acquired_entity: Tokyo
entity_role: destination_city
target_site: Eventbrite
completed_subgoal: identify_city
pending_subgoal: search_events
phase: cross_site_handoff
```

它可以与以下表面不相似任务建立关联：

```text
新闻网站找出获奖导演
→ IMDb 查询该导演的作品
```

二者共享的是 source-to-target entity handoff，而不是 Tokyo 或 Eventbrite 等词汇。

## 5.2 Multimodal-Mind2Web：约束满足与阶段状态

真实 TikTok Music 任务依次设置：

```text
Region       = Andorra
Use case     = TikTok Series
Genre        = Reggae
Mood         = Romantic
Artist       = BCD Studio
Submit       = Search
```

每一步的 situation 不同：

```text
Step 0：所有约束未设置
Step 2：Region 已完成，Use case 待设置
Step 4：Region/Use case/Genre 已完成，Mood 待设置
Step 5：类别约束完成，Artist 待输入
Step 6：全部约束满足，应提交
```

Situation 表示：

```yaml
type: ConstraintProgress
phase: form_filling
completed_constraints:
  - region=Andorra
  - use_case=TikTok Series
  - genre=Reggae
pending_constraints:
  - mood=Romantic
  - artist=BCD Studio
next_operation: CLICK_MOOD
```

这可以避免检索到“任务相似，但执行阶段不同”的历史轨迹。

## 5.3 WebVoyager：候选核验与冲突约束

真实任务：

```text
寻找 vegetarian lasagna：
reviews > 100
rating >= 4.5
suitable for 6 people
```

参考候选：

```text
Vegetarian Four Cheese Lasagna
rating = 4.6
reviews = 181
servings = 8
```

Situation：

```yaml
type: ConstraintCheck
phase: verify_candidate
candidate: Vegetarian Four Cheese Lasagna
satisfied:
  - vegetarian=true
  - rating=4.6>=4.5
  - reviews=181>100
conflicting:
  - servings=8
  - requested_servings=6
pending_decision:
  - adjust_quantities
  - or_search_another_candidate
```

真正有用的历史经验不是“如何搜索 lasagna”，而是：

> 候选满足大部分约束，但仍有一个冲突约束时，应继续搜索、调整结果还是明确说明差异。

## 5.4 GUI-Odyssey：系统设置 detour

真实 Todoist episode：

```text
打开 Settings
→ 搜索 Todoist
→ 修改通知状态
→ 返回 Home
→ 重新打开 Todoist
```

Situation：

```yaml
type: SystemSettingDetour
source_app: Todoist
detour_app: Android Settings
target_entity: Todoist notification setting
completed_subgoal: notification_modified
pending_subgoal: reopen_Todoist
transition: Settings_to_Home_to_Todoist
```

可关联的任务：

```text
修改 Spotify 权限 → 重开 Spotify
清理 Chrome 缓存 → 重开 Chrome
修改 Triller 通知 → 重开 Triller
```

这些任务表面内容不同，但共享：

\[
\text{App}
\rightarrow
\text{System-setting detour}
\rightarrow
\text{Return to app}.
\]

## 5.5 OSWorld：桌面工作流阶段

Chrome PDF 任务：

```text
打开 Print
→ Save to PDF
→ Margins=None
→ 保存到 Desktop
→ evaluator 检查 PDF
```

可区分：

```yaml
- phase: open_print_dialog
- phase: configure_destination
- phase: configure_margins
- phase: file_picker
- phase: verify_output
```

即使截图都来自 Chrome，不同阶段需要检索的历史经验也完全不同。

---

# 6. 两层记忆结构

## 6.1 Situation layer：用于索引与检索

Situation node 可包含：

```text
Goal
Application / Website
Page State
Task Phase
Active Subgoal
Completed Subgoal
Pending Subgoal
Known Entity
Satisfied Constraint
Unmet Constraint
Failure Mode
Expected Outcome
```

Situation layer 不一定保存完整截图，只保存足以决定记忆可达性的抽象状态。

## 6.2 Evidence layer：用于 Agent 决策

Evidence node 保存：

```text
原始 screenshot
ROI
页面文字/DOM/accessibility evidence
动作
动作结果
trajectory prefix
成功/失败 outcome
时间戳
来源文件
```

检索时先匹配 situation，最终仍返回可核验的原始 evidence。

CoMEM Q-Former 压缩的是 evidence，而不是 situation label。

---

# 7. 超图定义

普通 graph edge 通常只能表达两个节点之间的关系。GUI situation 往往需要多个变量联合成立，因此可使用 hyperedge。

## 7.1 信息交接超边

\[
\operatorname{Handoff}
(
\text{source},
\text{entity},
\text{target},
\text{pending subgoal}
).
\]

```yaml
type: Handoff
source: Wikipedia
entity: Tokyo
entity_role: city
target: Eventbrite
pending_subgoal: search_events
```

## 7.2 约束核验超边

\[
\operatorname{ConstraintCheck}
(
\text{candidate},
\text{satisfied},
\text{unmet},
\text{decision}
).
\]

```yaml
type: ConstraintCheck
candidate: Recipe_A
satisfied:
  - vegetarian
  - rating>=4.5
  - reviews>100
unmet:
  - serves_6
decision: adjust_or_continue_search
```

## 7.3 阶段转换超边

\[
\operatorname{PhaseTransition}
(
\text{old phase},
\text{action},
\text{new state},
\text{preserved information}
).
\]

```yaml
type: PhaseTransition
old_phase: discover_destination
action: switch_app
new_state: ride_hailing_destination_form
preserved_information: destination_name
```

## 7.4 错误恢复超边

\[
\operatorname{Recovery}
(
\text{failed state},
\text{failed action},
\text{cause},
\text{corrective action},
\text{outcome}
).
\]

```yaml
type: Recovery
failed_state: empty_search_results
failed_action: repeated_same_query
cause: over_specific_query
corrective_action: simplify_query
outcome: success
```

---

# 8. Situation 表示

在时间步 \(t\)，只允许使用当前及历史可观察信息：

\[
s_t=
f_{\phi}
\left(
q,
I_{\le t},
a_{<t},
o_{\le t},
u_t
\right),
\]

其中：

- \(q\)：任务指令；
- \(I_{\le t}\)：截至当前的截图/ROI；
- \(a_{<t}\)：历史动作；
- \(o_{\le t}\)：动作结果与页面变化；
- \(u_t\)：当前 URL、App、DOM 或 accessibility state。

结构化 situation：

```json
{
  "goal": "...",
  "app": "...",
  "page_state": "...",
  "phase": "...",
  "active_subgoal": "...",
  "completed_subgoals": [],
  "pending_subgoals": [],
  "known_entities": {},
  "satisfied_constraints": [],
  "unmet_constraints": [],
  "last_action": "...",
  "last_action_result": "...",
  "failure_mode": null,
  "expected_next_outcome": "..."
}
```

重要约束：

> Situation extractor 只能读取 trajectory prefix，不能读取未来动作、最终答案或完整成功轨迹。

否则会产生 future-information leakage。

---

# 9. 检索算法

## 9.1 原始 CoMEM

\[
S_{\text{clip}}(q,\tau_i)
=
\cos
\left(
k_q,k_i
\right).
\]

## 9.2 Situation-based retrieval

可定义：

\[
S_{\text{sit}}(s_t,s_i)
=
w_p S_{\text{phase}}
+w_g S_{\text{goal}}
+w_d S_{\text{dependency}}
+w_c S_{\text{constraint}}
+w_f S_{\text{failure}}.
\]

## 9.3 推荐的混合检索

第一版不建议完全删除 CLIP：

\[
S(\tau_i)
=
\alpha S_{\text{sit}}
+\beta S_{\text{clip}}
+\gamma S_{\text{outcome}}
+\delta S_{\text{phase}}.
\]

流程：

```text
Situation hypergraph candidate generation
    ↓
CLIP / learned scorer reranking
    ↓
diversity and outcome filtering
    ↓
top-K raw trajectories
```

这样更容易判断性能提升究竟来自 situation、视觉相似性还是二者互补。

---

# 10. 与 CoMEM 代码的结合点

## 10.1 保留部分

可以保持不变：

- trajectory JSON 与截图格式；
- CoMEM memory bank；
- Qwen2.5-VL trajectory encoder；
- 8-query Q-Former；
- continuous memory token 注入；
- frozen `model_inf`；
- next-action generation loss。

## 10.2 替换/新增部分

新增：

```text
SituationExtractor
SituationMetadataStore
HypergraphIndex
SituationRetriever
HybridReranker
RetrievalAuditLogger
```

替换：

```text
Memory.retrieve_similar_conversations()
```

或在其前增加 candidate generation：

```python
current_situation = situation_extractor(
    intent=intent,
    screenshot=current_image,
    action_history=action_history,
    page_state=page_state,
)

candidates = hypergraph.retrieve(current_situation)

ranked = hybrid_reranker(
    current_situation=current_situation,
    candidates=candidates,
    clip_query=clip_query,
)

selected_trajectories = ranked[:top_k]
```

后续仍调用原 CoMEM：

```text
selected trajectories
→ knowledge_processor
→ Q-Former
→ 8×K continuous vectors
→ model input embeddings
```

---

# 11. Situation 标签如何获得

## 11.1 规则抽取

可直接利用：

- 当前 URL/App；
- `procedure`；
- action type；
- page transition；
- selected filters；
- input values；
- success/failure；
- first error step。

规则适合高精度字段：

```text
app/site
action type
phase transition
completed field
failure/retry
```

## 11.2 MLLM 抽取

输入：

```text
任务
trajectory prefix
当前 screenshot
最近动作与结果
```

输出固定 schema 的 situation JSON。

需要：

- schema validation；
- 枚举约束；
- 去除未来信息；
- 人工抽样复核；
- 多模型一致性检查。

## 11.3 弱监督与对比学习

Positive pair：

```text
surface 不同，但 situation schema 匹配
```

Hard negative：

```text
任务/App 相似，但 phase 或 pending dependency 不同
```

训练目标：

\[
\mathcal L_{\text{sit}}
=
-\log
\frac{
\exp(\operatorname{sim}(s,s^+)/\tau)
}{
\exp(\operatorname{sim}(s,s^+)/\tau)
+
\sum_j
\exp(\operatorname{sim}(s,s_j^-)/\tau)
}.
\]

---

# 12. 数据集选择

## 第一阶段：MMInA + Multimodal-Mind2Web

优点：

- 已被 CoMEM 使用；
- 无需先接入 Android/VM；
- MMInA 有多跳 task/procedure；
- Mind2Web 有 screenshot、HTML 和专家动作；
- 容易抽取 search/filter/verify/submit phases。

建议先研究：

1. 跨网站信息交接；
2. 约束逐步满足；
3. 页面/任务阶段转换；
4. 错误恢复。

## 第二阶段：GUI-Odyssey

优点：

- 跨 App；
- 完整专家动作；
- App switch 与系统设置 detour 丰富；
- 最能证明 situation association。

代价：

- 当前 CoMEM 仓库没有 Android runner；
- 需要集成官方 AMS evaluator；
- 需要适配 continuous memory 到 mobile Agent。

## 第三阶段：OSWorld

优点：

- 真实桌面工作流；
- final-state evaluator；
- 支持不同正确操作路径；
- 适合验证长流程状态记忆。

代价：

- VM 环境复杂；
- App 多；
- 任务初始化和 evaluator 重；
- 当前 CoMEM 无 OSWorld adapter。

---

# 13. 关键评测设计

## 13.1 为什么原 benchmark split 不够

随机划分中，memory task 与 test task 可能共享：

- 相同网站；
- 相同 App；
- 相似措辞；
- 相似实体；
- 相似 screenshot。

这种设置即使成功，也不能证明 associative retrieval。

## 13.2 Surface-disjoint positive

要求底层 situation 相同，但：

- App/网站不同；
- 实体不同；
- 指令措辞不同；
- screenshot 差异大；
- 具体操作控件不同。

例如：

```text
Memory:
Wikipedia → Eventbrite
acquired entity = Tokyo

Query:
News website → IMDb
acquired entity = director name
```

## 13.3 Hard negative

表面相似但 situation 不同：

```text
同为 Google Maps：

A：尚未找到目标地点；
B：已经找到地点，正在切换到 Lyft。
```

若检索器只看 task/image similarity，容易混淆；situation retriever 应明确区分。

## 13.4 Counterfactual transfer

从真实轨迹构造：

```text
History:
安装 Triller → Settings 关闭通知 → 重开 Triller

Test:
安装另一 App → Settings 修改权限 → 重开该 App
```

只替换表面实体，保留任务结构，用于测试 situation transfer。

---

# 14. Baseline 与消融

至少比较：

1. No memory；
2. Random memory；
3. BM25 task-text retrieval；
4. CLIP text-only；
5. CLIP image-only；
6. CoMEM multimodal CLIP + FAISS；
7. Situation-only retrieval；
8. CLIP candidate + graph expansion；
9. Situation candidate + CLIP reranking；
10. Oracle situation retrieval。

关键消融：

- 去掉 phase；
- 去掉 pending dependency；
- 去掉 constraint state；
- 去掉 outcome；
- 去掉 failure/recovery；
- graph vs hypergraph；
- pairwise edges vs joint hyperedges；
- structured situation vs learned embedding；
- success-only vs success+failure memory；
- top-\(K\)；
- memory bank size。

为了控制变量，所有 retrieval baseline 应使用同一个：

- memory bank；
- top-\(K\)；
- CoMEM compressor；
- backbone；
- action generator；
- evaluator。

---

# 15. 指标

## 检索层

### Situation Recall@K

\[
\operatorname{SitRecall@K}
=
\frac{
\#\{\text{query 的正确 situation 进入 top-K}\}
}{
\#\{\text{queries}\}
}.
\]

### Evidence Recall@K

关键历史证据是否被召回。

### MRR

\[
\operatorname{MRR}
=
\frac1N
\sum_{i=1}^{N}
\frac1{\operatorname{rank}_i}.
\]

### False Retrieval Rate

召回表面相似但 situation 错误的比例。

## Agent 层

- Task Success Rate；
- Action Accuracy / AMS；
- Cross-site Transfer Success；
- Cross-App Transfer Success；
- Error Recovery Success；
- Memory Contribution Rate。

## 成本层

- retrieval latency；
- graph traversal latency；
- reranking latency；
- Q-Former compression latency；
- prompt/continuous token 数；
- end-to-end wall-clock；
- GPU memory；
- index storage。

---

# 16. Memory Contribution 审计

仅召回正确 memory 不代表 Agent 真正使用了它。

每个任务保存：

```json
{
  "query_situation": {},
  "activated_hyperedges": [],
  "retrieved_trajectory_ids": [],
  "retrieval_scores": {
    "situation": [],
    "clip": [],
    "hybrid": []
  },
  "used_evidence": [],
  "final_success": true,
  "counterfactual_without_memory": false
}
```

可定义：

\[
\operatorname{MemoryContributionRate}
=
P
\left(
\text{with memory succeeds}
\land
\text{without retrieved evidence fails}
\right).
\]

还可以进行：

- 删除单条 memory；
- 替换为 hard negative；
- 打乱 situation edge；
- 保留相同 soft-token 长度但使用随机向量。

这样才能证明成功来自正确联想，而不是额外 token 或随机提示效应。

---

# 17. 风险与防止错误结论

## 17.1 Situation 标签泄漏

不能使用：

- 最终答案；
- future screenshots；
- 后续成功动作；
- evaluator hidden reference；
- MMInA `eval.reference_answers`。

只能使用当前 trajectory prefix。

## 17.2 数据标注噪声

CoMEM 自动轨迹包含：

- judge 误判；
- 无效截图；
- 重复动作；
- 动态网页变化；
- 解析残留字段。

需要 schema validation 和人工抽检。

## 17.3 超图不等于创新

如果流程仍然是：

```text
CLIP top-k seed
→ graph neighbor expansion
```

则初始 similarity 选错区域时仍无法恢复。

必须证明 situation 本身能够建立新的可达性。

## 17.4 任务文本捷径

若 positive pair 共享大量词汇，模型可能只是学习 task similarity。

Surface-disjoint split 必须控制：

- lexical overlap；
- app overlap；
- entity overlap；
- screenshot similarity。

## 17.5 动态网页不可比

MMInA/WebVoyager 应记录：

- 访问时间；
- blocked/captcha；
- 页面版本；
- 无效任务；
- judge revision。

---

# 18. 最小可行实验

## Phase 1：离线检索验证

数据：

- Mind2Web expert trajectories；
- MMInA/CoMEM 成功轨迹前缀。

步骤：

1. 定义 4 类 situation schema；
2. 规则 + MLLM 抽取标签；
3. 构造 surface-disjoint positive 和 hard negative；
4. 比较 CLIP 与 situation retrieval；
5. 报 Recall@K、MRR、False Retrieval Rate。

这一阶段无需训练或运行完整 CoMEM。

## Phase 2：接入 CoMEM

1. 保持 Q-Former、backbone、memory bank 不变；
2. 只替换 retriever；
3. 固定 top-\(K=3\)；
4. 在 MMInA Wikipedia/Shopping 小子集运行；
5. 比较 task success 与 retrieval quality。

## Phase 3：规模与泛化

1. 增大 memory bank；
2. top-\(K=3,10,50\)；
3. 加入 failure/recovery memory；
4. 测试 surface-disjoint transfer；
5. 接入 GUI-Odyssey。

## Phase 4：完整 OOD

- GUI-Odyssey cross-app；
- OSWorld final-state tasks；
- 比较 web memory 向 mobile/desktop 的迁移。

---

# 19. 预期论文贡献

可以形成四点：

1. **Problem formulation**  
   定义 GUI Agent 中 surface-disconnected associative retrieval。

2. **Situation-indexed memory**  
   使用 goal、phase、dependency、constraint、failure 等联合状态建立记忆索引。

3. **Evidence-grounded continuous injection**  
   检索索引与注入证据解耦，并复用 CoMEM continuous compressor。

4. **Situation-disjoint benchmark**  
   构造表面不相似但情境相同的 positive pairs，以及表面相似但阶段不同的 hard negatives。

---

# 20. 结论

该方向与 CoMEM 不是竞争关系，而是自然的前后级组合：

\[
\boxed{
\text{CoMEM}
=
\text{How to compress and inject retrieved trajectories}
}
\]

\[
\boxed{
\text{Situation Memory}
=
\text{How to retrieve the right trajectory}
}
\]

最合理的系统是：

\[
\boxed{
\text{Situation-indexed retrieval}
\rightarrow
\text{evidence-grounded trajectory}
\rightarrow
\text{CoMEM continuous compression}
}
\]

研究价值不在于“使用超图”，而在于证明：

> 当当前 GUI 与历史轨迹在任务文字、应用和视觉上都不相似时，Agent 仍然能根据共同 situation 找到并利用真正有帮助的历史证据。

---

# 参考与核验说明

本研究构想结合：

- `docs/联想记忆检索研究.pdf` 中的二次调研；
- CoMEM 论文 PDF/TeX；
- 当前 CoMEM-Agent 代码；
- MMInA、Multimodal-Mind2Web、WebVoyager、GUI-Odyssey、OSWorld 的公开样例和 evaluator。

注意：`联想记忆检索研究.pdf` 是 ChatGPT 调研对话副本，不是原始论文。对 T-Mem、Mem-W、HyMEM、MementoGUI、Agent Workflow Memory 等工作的具体技术主张，在正式论文中引用前必须回到原论文和官方代码逐项核验。

