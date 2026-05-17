# VGPCC（Verifier-Guided Personalized Context Compression）

本文档基于代码实现说明 `pipeline.py` 的模型架构与训练/评估流程，重点解释各模块（verifier / teacher / judger / repairer 等）的职责与数据流。

## 架构总览 🧩

系统目标：在给定 token 预算下压缩用户长期上下文，并通过 verifier 反馈进行动态修复与再压缩，尽量保留个性化效用。

```
flowchart TB
    A[用户请求 Query] --> B[MemoryIndex 构建/加载]
    A --> C[BaseCompressor 压缩]
    B --> C
    C --> D[Online Verifier]
    D -->|不足| E[Targeted Repairer]
    D -->|充分| F[Downstream Executor]
    E --> F

    subgraph 离线训练
      T1[Teacher Verifier (全上下文)] --> T2[Utility Loss Judger]
      C --> T1
      C --> T2
      T1 --> T3[训练样本筛选]
      T2 --> T3
      T3 --> T4[Verifier 微调]
    end
```

## 核心模块说明

### 1. MemoryIndex & SummaryRefiner
- **MemoryIndexLoader**：加载 `summary_cache.json` 或细化后的 `summary_cache_vgpcc.json`，形成可检索的记忆索引（按 id、tag、type 组织）。
- **SummaryRefiner**：可选，用 LLM 把原始 summary 转成结构化记忆条目（含 type / importance / tag 等字段）。

### 2. BaseCompressor（压缩器）
- 输入：`profile`（完整上下文）、`memory_index`、`query`、`token_budget`。
- 输出：
  - `compressed_context`
  - `trajectory`（记录 kept_ids、丢弃的高重要性条目、覆盖统计、压缩率等）
- 支持 **LLM 压缩** 或 **启发式压缩** 两种模式。

### 3. Online Verifier（在线验证器）
- 只看 **压缩上下文 + 记忆索引统计 + 轨迹信息**，不看完整原文。
- 输出 `VerificationResult`：
  - `is_sufficient` / `sufficiency_score`
  - `missing_evidence_type`（Style/Recent_Preference/Hard_Constraint/Task_Memory/Preference_Conflict/None）
  - `predicted_utility_loss`
  - `diagnostic_reasoning`

### 4. Feedback Controller（压缩反馈）
- 如果 verifier 认为不足：
  - 增大 token 预算（`feedback_increase_factor`）
  - 或只针对 `missing_evidence_type` 进行再次压缩
- 得到新的 `compressed_context` 与 `trajectory`

### 5. Targeted Repairer（修复器）
- 当 verifier 判定缺失某类证据时：
  - 从 `memory_index` 中补回对应类型的若干条 tag
  - 形成 `final_context`

### 6. Downstream Executor（下游任务执行）
- 使用 `Query + final_context` 进行 LaMP-3 任务（偏好判断/生成）。

### 7. Teacher Verifier & Judger（离线训练专用）
- **Teacher Verifier**：允许访问完整上下文，输出 sufficiency 监督信号。
- **Utility Loss Judger**：比较 full-context 与 compressed-context 的输出差异，估计 `utility_loss`。
- 两者用于构造训练数据并筛选高质量样本。

## 训练流程（`--mode train` + `--mode train_verifier`）

训练分为两步：

### 训练中“会更新/不会更新”的参数

- **会更新（仅在 `--mode train_verifier`）**
  - `Online Verifier` 的 **模型参数（weights）**：由 `Trainer` 对 `verifier_model.model` 进行微调并保存到 `verifier_out/`。
  - `verifier_model` 的 tokenizer **配置文件**会随权重一并保存（不包含参数学习，仅保存配置）。

- **不会更新（训练期间固定）**
  - **BaseCompressor** 的算法逻辑与阈值：只依赖 `config.yaml` 中的 `pipeline.*` 设置。
  - **Downstream Executor / Teacher Verifier / Utility Loss Judger** 使用的模型参数：仅用于前向推理生成监督信号，不参与反向更新。
  - **MemoryIndex / SummaryRefiner**：只做加载与（可选）摘要结构化，不会被训练优化。
  - **训练过滤阈值**（`training.*`）：仅用于筛选样本，不会被学习。

1. **训练样本构造（collect_train_data）**
   - 对每条样本：
     1) 压缩 → Online Verifier → 反馈再压缩
     2) Teacher Verifier 读取完整上下文，给出 `is_sufficient` 与 `sufficiency_score`
     3) Judger 比较 full vs compressed 的下游输出，估计 `utility_loss`
     4) 过滤质量低的样本（阈值由 `training` 配置控制）
     5) 可选 adversarial attack：删除某类证据，构造缺失类型标签
   - 输出：`vgpcc_train.jsonl`

2. **Verifier 微调（train_verifier）**
  - 使用 `VerifierTrainDataset` 封装 JSONL
  - 目标是让 Online Verifier 学会输出与 Teacher 一致的 sufficiency/缺失类型
  - **仅更新 verifier 权重**，不会影响压缩器或下游主模型

## 评估流程（`--mode eval`）

评估的主流程如下：

1. 读取样本（query / profile / gold）
2. 压缩上下文 + 在线验证 + 反馈再压缩
3. 若 verifier 判定缺失证据，则触发 Targeted Repair
4. 下游任务执行（得到 output）
5. 若有 gold，则用 **exact match** 作为 accuracy
6. 输出日志与统计：`log_path` 与 `log_path.summary.json`

## 关键配置说明（`config.yaml`）

- `models.qwen3_8b`：下游任务与压缩使用的主模型
- `models.verifier_base` / `verifier_finetuned`：verifier 基座与微调权重
- `summaries.refine_with_llm`：是否将 summary 细化成结构化记忆
- `pipeline.token_budget`：压缩 token 预算
- `pipeline.feedback_*`：反馈阈值与预算扩大策略
- `training.*`：样本过滤门槛（sufficiency 与 utility loss）

## 输出产物

- **训练数据**：`vgpcc_train.jsonl`
- **日志**：`logs/*.log.jsonl` 与 `*.summary.json`
- **细化记忆**：`data/summary_cache_vgpcc.json`
- **Verifier 微调权重**：`verifier_ft/`

## 小结

本 pipeline 通过「压缩→验证→修复→再压缩」闭环，把 verifier 作为动态反馈信号，在有限 token 预算内尽可能保留个性化证据，同时提供 offline teacher & judge 监督用于 verifier 训练。
