import argparse
import json
import os
import sys
from typing import Iterable

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import append_method_tag, get_doc_range, get_method_tag, get_range_tag, read_json_file_as_df


def read_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_predictions(file_path: str) -> list[dict]:
    lower_path = file_path.lower()
    if lower_path.endswith(".jsonl"):
        return read_jsonl(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dedupe_triples(records: Iterable[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in records:
        key = (item["title"], int(item["h_idx"]), int(item["t_idx"]), item["r"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "title": item["title"],
                "h_idx": int(item["h_idx"]),
                "t_idx": int(item["t_idx"]),
                "r": item["r"],
            }
        )
    return unique


def build_gold_triples(docs: list[dict]) -> set[tuple[str, int, int, str]]:
    gold = set()
    for doc in docs:
        title = doc["title"]
        for label in doc.get("labels", []):
            gold.add((title, int(label["h"]), int(label["t"]), label["r"]))
    return gold


def compute_scores(pred_triples: set[tuple[str, int, int, str]], gold_triples: set[tuple[str, int, int, str]]) -> dict:
    tp = len(pred_triples & gold_triples)
    fp = len(pred_triples - gold_triples)
    fn = len(gold_triples - pred_triples)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocRED predictions on a document subset.")
    parser.add_argument("--data-name", default=os.getenv("DATA_NAME", "dev"))
    parser.add_argument("--gold-path", default="../data/docred/dev.json")
    parser.add_argument("--pred-path", default="")
    parser.add_argument("--method-tag", default=os.getenv("METHOD_TAG", ""))
    parser.add_argument("--doc-start", type=int, default=None)
    parser.add_argument("--doc-end", type=int, default=None)
    args = parser.parse_args()

    gold_df = read_json_file_as_df(args.gold_path)
    total_len = len(gold_df)

    if args.doc_start is None or args.doc_end is None:
        doc_start, doc_end = get_doc_range(total_len)
    else:
        doc_start = max(0, min(args.doc_start, total_len))
        doc_end = max(doc_start, min(args.doc_end, total_len))

    range_tag = get_range_tag(doc_start, doc_end)

    pred_path = args.pred_path
    if not pred_path:
        method_tag = args.method_tag.strip() or get_method_tag()
        save_doc_name = append_method_tag(f"k20-docred_{range_tag}", method_tag)
        pred_path = f"../data/get_triplet_fact_judgement_label/{args.data_name}/docred_{args.data_name}_triplet_fact_judgement_{range_tag}_answer-{save_doc_name}.jsonl"

    gold_docs = gold_df.iloc[doc_start:doc_end].to_dict(orient="records")
    gold_titles = {doc["title"] for doc in gold_docs}
    gold_triples = build_gold_triples(gold_docs)

    pred_records = dedupe_triples(load_predictions(pred_path))
    in_subset_preds = [item for item in pred_records if item["title"] in gold_titles]
    out_of_subset_preds = [item for item in pred_records if item["title"] not in gold_titles]
    pred_triples = {
        (item["title"], int(item["h_idx"]), int(item["t_idx"]), item["r"])
        for item in in_subset_preds
    }

    metrics = compute_scores(pred_triples, gold_triples)

    print(f"pred_path: {pred_path}")
    print(f"gold_path: {args.gold_path}")
    print(f"subset_range: {range_tag}")
    print(f"gold_doc_count: {len(gold_docs)}")
    print(f"gold_relation_count: {len(gold_triples)}")
    print(f"prediction_count_in_subset: {len(pred_triples)}")
    print(f"prediction_count_out_of_subset: {len(out_of_subset_preds)}")
    print(f"tp: {metrics['tp']}")
    print(f"fp: {metrics['fp']}")
    print(f"fn: {metrics['fn']}")
    print(f"precision: {metrics['precision']:.6f}")
    print(f"recall: {metrics['recall']:.6f}")
    print(f"f1: {metrics['f1']:.6f}")


if __name__ == "__main__":
    main()
