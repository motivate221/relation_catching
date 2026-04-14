import argparse
import json
import os
import sys
from typing import Iterable

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import get_range_tag, read_json_file_as_df


def read_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def canonicalize_title(title: str) -> str:
    replacements = {
        "IBM Research 鈥?Brazil": "IBM Research – Brazil",
        "IBM Research 每 Brazil": "IBM Research – Brazil",
        "IBM Research ? Brazil": "IBM Research – Brazil",
    }
    return replacements.get(title, title)


def load_gold_subset(gold_path: str, doc_start: int, doc_end: int) -> tuple[set[tuple[str, int, int, str]], dict[str, list[dict]]]:
    gold_df = read_json_file_as_df(gold_path)
    subset_docs = gold_df.iloc[doc_start:doc_end].to_dict(orient="records")
    gold = set()
    docs_by_title = {}
    for doc in subset_docs:
        title = doc["title"]
        docs_by_title[title] = doc
        for label in doc.get("labels", []):
            gold.add((title, int(label["h"]), int(label["t"]), label["r"]))
    return gold, docs_by_title


def load_rel_info(rel_info_path: str) -> tuple[dict[str, str], dict[str, str]]:
    with open(rel_info_path, "r", encoding="utf-8") as f:
        rel_info = eval(f.read())
    reverse_rel_info = {v: k for k, v in rel_info.items()}
    return rel_info, reverse_rel_info


def build_stage1_pairs(file_path: str) -> set[tuple[str, int, int]]:
    rows = read_jsonl(file_path)
    return {
        (canonicalize_title(row["title"]), int(row["h_idx"]), int(row["t_idx"]))
        for row in rows
    }


def build_stage4_retrieval_triples(file_path: str, rel_info: dict[str, str]) -> set[tuple[str, int, int, str]]:
    rows = read_jsonl(file_path)
    triples = set()
    for row in rows:
        left = row[0]
        title = canonicalize_title(left["title"])
        h_idx = int(left["entity_h_id"])
        t_idx = int(left["entity_t_id"])
        for rel_name, _payload in row[1]:
            if rel_name in rel_info:
                rel_id = rel_info[rel_name]
                triples.add((title, h_idx, t_idx, rel_id))
                triples.add((title, t_idx, h_idx, rel_id))
    return triples


def build_stage4_rerank_triples(file_path: str, rel_info: dict[str, str]) -> set[tuple[str, int, int, str]]:
    rows = read_jsonl(file_path)
    triples = set()
    for row in rows:
        title = canonicalize_title(row["title"])
        h_idx = int(row["entity_h_id"])
        t_idx = int(row["entity_t_id"])
        for rel_name in row.get("top_relations", []):
            if rel_name in rel_info:
                rel_id = rel_info[rel_name]
                triples.add((title, h_idx, t_idx, rel_id))
                triples.add((title, t_idx, h_idx, rel_id))
    return triples


def build_prediction_triples(file_path: str) -> set[tuple[str, int, int, str]]:
    rows = read_jsonl(file_path)
    return {
        (canonicalize_title(row["title"]), int(row["h_idx"]), int(row["t_idx"]), row["r"])
        for row in rows
    }


def summarize_stage(name: str, gold: set[tuple[str, int, int, str]], stage_set: set, pair_only: bool = False) -> dict:
    if pair_only:
        kept = {
            triple for triple in gold
            if (triple[0], triple[1], triple[2]) in stage_set
        }
    else:
        kept = gold & stage_set
    return {
        "name": name,
        "kept_count": len(kept),
        "dropped_count": len(gold) - len(kept),
        "kept": kept,
    }


def print_summary(section_name: str, summaries: list[dict], total_gold: int) -> None:
    print(f"=== {section_name} ===")
    print(f"gold_total: {total_gold}")
    for summary in summaries:
        kept = summary["kept_count"]
        dropped = summary["dropped_count"]
        ratio = kept / total_gold if total_gold else 0.0
        print(f"{summary['name']}: kept={kept} dropped={dropped} keep_ratio={ratio:.4f}")
    print("")


def print_missing_examples(stage_name: str, gold: set[tuple[str, int, int, str]], kept: set[tuple[str, int, int, str]], limit: int = 10) -> None:
    missing = sorted(gold - kept)
    print(f"{stage_name} missing examples (up to {limit}):")
    for item in missing[:limit]:
        print(item)
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze relation dropout across DocRE pipeline stages.")
    parser.add_argument("--data-name", default="dev")
    parser.add_argument("--doc-start", type=int, required=True)
    parser.add_argument("--doc-end", type=int, required=True)
    args = parser.parse_args()

    range_tag = get_range_tag(args.doc_start, args.doc_end)
    base_dir = os.path.dirname(os.path.dirname(__file__))

    gold_path = os.path.join(base_dir, "data", "docred", f"{args.data_name}.json")
    rel_info_path = os.path.join(base_dir, "data", "docred", "rel_info.json")

    gold, _docs = load_gold_subset(gold_path, args.doc_start, args.doc_end)
    rel_info, _reverse_rel_info = load_rel_info(rel_info_path)

    stage1_pairs = build_stage1_pairs(
        os.path.join(base_dir, "data", "get_entity_pair_selection_label", args.data_name, f"docred_{args.data_name}_entity_pair_selection_{range_tag}_answer-01.jsonl")
    )
    stage4_baseline = build_stage4_retrieval_triples(
        os.path.join(base_dir, "data", "retrieval_from_train", args.data_name, f"path-k20-docred_{range_tag}.jsonl"),
        rel_info,
    )
    stage4_rerank = build_stage4_rerank_triples(
        os.path.join(base_dir, "data", "retrieval_rerank", args.data_name, f"rerank-k20-docred_{range_tag}.jsonl"),
        rel_info,
    )
    stage5_baseline = build_prediction_triples(
        os.path.join(base_dir, "data", "get_multiple_choice_label", args.data_name, f"docred_{args.data_name}_multiple_choice_path-k20-docred_{range_tag}_baseline_answer.jsonl")
    )
    stage5_rerank = build_prediction_triples(
        os.path.join(base_dir, "data", "get_multiple_choice_label", args.data_name, f"docred_{args.data_name}_multiple_choice_path-k20-docred_{range_tag}_rerank_answer.jsonl")
    )
    stage6_baseline = build_prediction_triples(
        os.path.join(base_dir, "data", "get_triplet_fact_judgement_label", args.data_name, f"docred_{args.data_name}_triplet_fact_judgement_{range_tag}_answer-k20-docred_{range_tag}_baseline.jsonl")
    )
    stage6_rerank = build_prediction_triples(
        os.path.join(base_dir, "data", "get_triplet_fact_judgement_label", args.data_name, f"docred_{args.data_name}_triplet_fact_judgement_{range_tag}_answer-k20-docred_{range_tag}_rerank.jsonl")
    )

    baseline_summaries = [
        summarize_stage("stage1_pair_selection", gold, stage1_pairs, pair_only=True),
        summarize_stage("stage4_retrieval", gold, stage4_baseline),
        summarize_stage("stage5_multiple_choice_baseline", gold, stage5_baseline),
        summarize_stage("stage6_fact_judgement_baseline", gold, stage6_baseline),
    ]
    rerank_summaries = [
        summarize_stage("stage1_pair_selection", gold, stage1_pairs, pair_only=True),
        summarize_stage("stage4_retrieval_top3_rerank", gold, stage4_rerank),
        summarize_stage("stage5_multiple_choice_rerank", gold, stage5_rerank),
        summarize_stage("stage6_fact_judgement_rerank", gold, stage6_rerank),
    ]

    print_summary("Baseline Flow", baseline_summaries, len(gold))
    print_summary("Rerank Flow", rerank_summaries, len(gold))

    print_missing_examples("Baseline final", gold, baseline_summaries[-1]["kept"])
    print_missing_examples("Rerank final", gold, rerank_summaries[-1]["kept"])


if __name__ == "__main__":
    main()
