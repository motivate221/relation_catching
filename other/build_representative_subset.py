import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_doc_stats(doc):
    rels = doc.get("labels", [])
    rel_count = len(rels)
    entity_count = len(doc.get("vertexSet", []))
    rel_types = set(label["r"] for label in rels)
    return rel_count, entity_count, rel_types


def score_subset(
    subset_indices,
    docs,
    all_rel_types,
    rare_rel_types,
    global_rel_mean,
    global_ent_mean,
    target_size,
):
    subset_rel_counts = []
    subset_ent_counts = []
    subset_rel_types = set()
    subset_rare_rel_types = set()

    for idx in subset_indices:
        rel_count, ent_count, rel_types = get_doc_stats(docs[idx])
        subset_rel_counts.append(rel_count)
        subset_ent_counts.append(ent_count)
        subset_rel_types |= rel_types
        subset_rare_rel_types |= (rel_types & rare_rel_types)

    rel_sum = sum(subset_rel_counts)
    rel_mean = rel_sum / max(len(subset_rel_counts), 1)
    ent_mean = sum(subset_ent_counts) / max(len(subset_ent_counts), 1)

    rel_cov = len(subset_rel_types) / max(len(all_rel_types), 1)
    rare_rel_cov = len(subset_rare_rel_types) / max(len(rare_rel_types), 1) if rare_rel_types else 1.0

    rel_mean_gap = abs(rel_mean - global_rel_mean) / max(global_rel_mean, 1e-9)
    ent_mean_gap = abs(ent_mean - global_ent_mean) / max(global_ent_mean, 1e-9)

    target_rel_sum = target_size * global_rel_mean
    rel_sum_fill = min(rel_sum / max(target_rel_sum, 1e-9), 1.0)

    score = (
        3.0 * rel_cov
        + 1.5 * rare_rel_cov
        + 0.5 * rel_sum_fill
        - 1.2 * rel_mean_gap
        - 0.8 * ent_mean_gap
    )

    stats = {
        "relation_coverage": rel_cov,
        "rare_relation_coverage": rare_rel_cov,
        "relation_sum": rel_sum,
        "relation_mean": rel_mean,
        "entity_mean": ent_mean,
        "relation_mean_gap": rel_mean_gap,
        "entity_mean_gap": ent_mean_gap,
        "score": score,
    }
    return score, stats


def build_subset(
    docs,
    size,
    seed,
    iterations,
    rare_freq_threshold,
):
    if size <= 0 or size > len(docs):
        raise ValueError(f"Invalid size={size}. Must be in [1, {len(docs)}].")

    rel_counter = Counter()
    all_rel_types = set()
    rel_counts = []
    ent_counts = []
    for doc in docs:
        rel_count, ent_count, rel_types = get_doc_stats(doc)
        rel_counts.append(rel_count)
        ent_counts.append(ent_count)
        all_rel_types |= rel_types
        for r in rel_types:
            rel_counter[r] += 1

    rare_rel_types = {r for r, c in rel_counter.items() if c <= rare_freq_threshold}
    global_rel_mean = sum(rel_counts) / len(rel_counts)
    global_ent_mean = sum(ent_counts) / len(ent_counts)

    rng = random.Random(seed)
    indices = list(range(len(docs)))

    best_subset = None
    best_stats = None
    best_score = float("-inf")

    for _ in range(iterations):
        subset = rng.sample(indices, size)
        score, stats = score_subset(
            subset,
            docs,
            all_rel_types,
            rare_rel_types,
            global_rel_mean,
            global_ent_mean,
            size,
        )
        if score > best_score:
            best_score = score
            best_subset = subset
            best_stats = stats

    best_subset = sorted(best_subset)
    subset_docs = [docs[i] for i in best_subset]

    return subset_docs, best_subset, best_stats, {
        "doc_count_full": len(docs),
        "relation_type_count_full": len(all_rel_types),
        "rare_relation_type_count_full": len(rare_rel_types),
        "global_relation_mean": global_rel_mean,
        "global_entity_mean": global_ent_mean,
        "rare_freq_threshold": rare_freq_threshold,
        "search_iterations": iterations,
        "random_seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Build a representative DocRED dev subset for stable small-sample evaluation.")
    parser.add_argument("--input", default="../data/docred/dev.json")
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--rare-freq-threshold", type=int, default=15)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    docs = load_json(input_path)

    subset_docs, subset_indices, subset_stats, meta = build_subset(
        docs=docs,
        size=args.size,
        seed=args.seed,
        iterations=args.iterations,
        rare_freq_threshold=args.rare_freq_threshold,
    )

    subset_name = args.name.strip() or f"dev_repr_{args.size}"
    output_json = input_path.parent / f"{subset_name}.json"
    manifest_json = input_path.parent / "subsets" / f"{subset_name}_manifest.json"

    save_json(output_json, subset_docs)
    save_json(
        manifest_json,
        {
            "subset_name": subset_name,
            "input_file": str(input_path),
            "output_file": str(output_json),
            "indices_in_original_dev": subset_indices,
            "titles": [doc.get("title", "") for doc in subset_docs],
            "subset_stats": subset_stats,
            "meta": meta,
        },
    )

    print(f"subset saved: {output_json}")
    print(f"manifest saved: {manifest_json}")
    print("subset_stats:")
    for k, v in subset_stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

