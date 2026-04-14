# 数据流转表（DocRED 主流程）

> 变量说明：`{data_name}` 一般为 `dev` / `train_annotated`，`{doc_name}` 一般为 `docred`，`{range_tag}` 如 `0-5` / `0-998`，`{method_tag}` 如 `baseline` / `rerank`。  
> 主流程脚本：`run_dev_0_20_pipeline.ps1`（阶段1~6）+ `run_train_annotated_cache_pipeline.ps1`（训练侧缓存）。

| 序号 | 阶段/模块 | 关键脚本 | 主要输入 | 主要输出 | 关键字段（核心） | 说明 |
|---|---|---|---|---|---|---|
| 0 | 原始数据 | `data/docred/*.json` | `dev.json` / `train_annotated.json` | 文档、实体、标注 | `title`, `sents`, `vertexSet`, `labels` | 全流程数据源 |
| 1 | 阶段1 实体对筛选（最终标签） | `1.entity_pair_selection/get_entity_pair_selection_label.py` | `../data/check_result_entity_pair_selection_jsonl/{data_name}/result_{doc_name}_{data_name}_entity_pair_selection_{range_tag}-01.jsonl` | `../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_{range_tag}_answer-01.jsonl` | `title`, `h_idx`, `t_idx`, `r` | 产出候选实体对（关系占位 `r=P1`） |
| 2 | 阶段2 实体信息抽取（汇总结果） | `2.entity_information/check_result_entity_information_jsonl.py` | `../data/entity_information_prompt/{data_name}/prompt_{doc_name}_{data_name}_entity_information_doc{range_tag}.jsonl` + run分片 | `../data/entity_information/{data_name}/result_{doc_name}_{data_name}_entity_information_{range_tag}.jsonl` | `title`, `entity`, `response` | 形成实体语义描述 |
| 3 | 阶段3 关系摘要生成（汇总结果） | `3.relation_summary/check_result_relation_summary_jsonl.py` | `../data/relation_summary_prompt/{data_name}/result_{doc_name}_{data_name}_doc{range_tag}.jsonl` + run分片 | `../data/check_result_relation_summary_jsonl/{data_name}/result_{doc_name}_{data_name}_relation_summary_{range_tag}.jsonl` | `title`, `entity_h_id`, `entity_t_id`, `label_rel`, `entities_description` | 后续检索与候选关系召回的语义基础 |
| 4 | 阶段4 向量化（Embedding） | `4.retrieval/get_embeddings.py` | 阶段2实体信息 + 阶段3关系摘要 | `../data/get_embeddings/{doc_name}_{data_name}_embeddings_{range_tag}.npy` | 向量矩阵（与摘要条目对齐） | 为近邻检索提供向量索引 |
| 5 | 阶段4 检索召回（Top-k） | `4.retrieval/retrieval_from_train-few.py` | 训练侧缓存（`train_annotated` 关系摘要+embedding）+ 当前 `{data_name}` embedding | `../data/retrieval_from_train/{data_name}/path-k20-{doc_name}_{range_tag}.jsonl` | `base_info`, `candidate_relations` | 从训练侧检索候选关系路径 |
| 6 | 阶段4.5 证据感知重排（可选） | `4.retrieval/evidence_relation_rerank.py` | 检索结果 + 阶段3关系摘要 + 类型约束 | `../data/retrieval_rerank/{data_name}/rerank-k20-{doc_name}_{range_tag}.jsonl` | `top_relations`, `prompt_rel`, `routing_decision`, `evidence_sentence_ids` | 创新模块：多信号融合重排 |
| 7 | 阶段5 多选精筛（最终标签） | `5.multiple_choice/get_multiple_choice_label.py` | `../data/check_result_multiple_choice_jsonl/{data_name}/result_{doc_name}_{data_name}_multiple_choice_path-k20-{doc_name}_{range_tag}_{method_tag}.jsonl` | `../data/get_multiple_choice_label/{data_name}/{doc_name}_{data_name}_multiple_choice_path-k20-{doc_name}_{range_tag}_{method_tag}_answer.jsonl` | `title`, `h_idx`, `t_idx`, `r` | 将候选关系压缩为更高置信关系 |
| 8 | 阶段6 事实判定（最终输出） | `6.triplet_fact_judgement/get_triplet_fact_judgement_label.py` | `../data/check_result_triplet_fact_judgement_jsonl/{data_name}/result_{doc_name}_{data_name}_triplet_fact_judgement_k20-{doc_name}_{range_tag}_{method_tag}.jsonl` | `../data/get_triplet_fact_judgement_label/{data_name}/{doc_name}_{data_name}_triplet_fact_judgement_{range_tag}_answer-k20-{doc_name}_{range_tag}_{method_tag}.jsonl` | `title`, `h_idx`, `t_idx`, `r` | 全流程最终三元组文件 |
| 9 | 评测 | `other/evaluate_subset.py` / `other/evaluation.py` | 阶段6最终输出 + `data/docred/dev.json` | 指标结果（控制台/记录文件） | `precision`, `recall`, `f1` | 支持 baseline 与 rerank 对照 |

## 训练侧缓存流（支撑阶段4检索）

| 流程 | 关键产物 | 作用 |
|---|---|---|
| `run_train_annotated_cache_pipeline.ps1` | `../data/check_result_relation_summary_jsonl/train_annotated/result_docred_train_annotated_relation_summary_0-3053.jsonl` | 训练侧语义关系摘要库 |
| `run_train_annotated_cache_pipeline.ps1` | `../data/get_embeddings/docred_train_annotated_embeddings_0-3053.npy` | 训练侧向量检索库 |

---

若用于论文/中期报告，建议将本表与“系统架构图”一起使用：  
1）架构图说明模块关系；2）数据流转表说明每一阶段的输入输出与文件落盘位置。

