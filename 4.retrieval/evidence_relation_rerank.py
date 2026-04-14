import json
import os
import sys
from typing import Any

from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import get_doc_range, get_range_tag, read_json_file_as_df
from rerank_config import (
    DIRECT_VERIFY_GAP,
    DIRECT_VERIFY_THRESHOLD,
    DISTANCE_WEIGHT,
    EVIDENCE_SENT_NUM,
    EVIDENCE_WEIGHT,
    RETRIEVAL_WEIGHT,
    SUMMARY_WEIGHT,
    TOPK_RETRIEVAL,
    TOPM_RERANK,
    TYPE_WEIGHT,
    USE_TYPE_CONSTRAINT,
)


def read_jsonl(file_path: str) -> list[Any]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_to_jsonl(data: list[dict[str, Any]], jsonl_file: str) -> None:
    output_dir = os.path.dirname(jsonl_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_file, "w", encoding="utf-8") as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii=False)
            jsonlfile.write("\n")


def read_list_text_file_to_list(file_path: str) -> list[Any]:
    text_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            text_data.append(eval(line))
    return text_data


def get_docid(title: str, df) -> int:
    for doc_id in range(len(df)):
        if title == df["title"][doc_id]:
            return doc_id
    raise ValueError(f"title not found: {title}")


def get_sentence_text(sentence_tokens: list[str]) -> str:
    return " ".join(sentence_tokens).strip()


def get_document_sentences(doc_id: int, df) -> list[str]:
    return [get_sentence_text(sentence) for sentence in df["sents"][doc_id]]


def get_entity_sentence_ids(doc_id: int, entity_id: int, df) -> list[int]:
    sent_ids = []
    for mention in df["vertexSet"][doc_id][entity_id]:
        sent_ids.append(mention["sent_id"])
    return sorted(set(sent_ids))


def compute_distance_score(doc_id: int, entity_h_id: int, entity_t_id: int, df) -> float:
    h_sent_ids = get_entity_sentence_ids(doc_id, entity_h_id, df)
    t_sent_ids = get_entity_sentence_ids(doc_id, entity_t_id, df)
    min_gap = min(abs(h_id - t_id) for h_id in h_sent_ids for t_id in t_sent_ids)
    return 1.0 / (1.0 + float(min_gap))


def build_rel_judge_dict(rel_judge_file_path: str) -> dict[str, list[tuple[str, str]]]:
    rel_judge_list = read_list_text_file_to_list(rel_judge_file_path)[0]
    rel_judge_dict: dict[str, list[tuple[str, str]]] = {}
    for row in rel_judge_list:
        fruits = row[0].split("_")
        rel_judge_dict.setdefault(fruits[0], []).append((fruits[1], fruits[2]))
    return rel_judge_dict


def rel_h_t_judge(
    rel: str,
    entity_h_id: int,
    entity_t_id: int,
    rel_judge_dict: dict[str, list[tuple[str, str]]],
    reverse_rel_info: dict[str, str],
    doc_id: int,
    df,
) -> bool:
    entity_h_type = df["vertexSet"][doc_id][entity_h_id][0]["type"]
    entity_t_type = df["vertexSet"][doc_id][entity_t_id][0]["type"]
    key = reverse_rel_info[rel]
    if key not in rel_judge_dict:
        return False
    for head_type, tail_type in rel_judge_dict[key]:
        if head_type == entity_h_type and tail_type == entity_t_type:
            return True
    return False


def collect_candidate_sentence_ids(doc_id: int, entity_h_id: int, entity_t_id: int, df) -> list[int]:
    h_sent_ids = get_entity_sentence_ids(doc_id, entity_h_id, df)
    t_sent_ids = get_entity_sentence_ids(doc_id, entity_t_id, df)
    overlap = sorted(set(h_sent_ids) & set(t_sent_ids))
    remaining = sorted((set(h_sent_ids) | set(t_sent_ids)) - set(overlap))
    return overlap + remaining


def select_evidence_sentences(
    doc_id: int,
    entity_h_id: int,
    entity_t_id: int,
    entities_description: str,
    df,
    model: SentenceTransformer,
    max_sent_num: int,
) -> tuple[list[int], list[str]]:
    all_sentences = get_document_sentences(doc_id, df)
    candidate_sent_ids = collect_candidate_sentence_ids(doc_id, entity_h_id, entity_t_id, df)
    selected_ids: list[int] = []

    for sent_id in candidate_sent_ids:
        if sent_id not in selected_ids:
            selected_ids.append(sent_id)
        if len(selected_ids) >= max_sent_num:
            break

    if len(selected_ids) < max_sent_num and all_sentences:
        summary_embedding = model.encode(entities_description, convert_to_tensor=True)
        sentence_embeddings = model.encode(all_sentences, convert_to_tensor=True)
        scores = util.cos_sim(summary_embedding, sentence_embeddings).reshape(-1).tolist()
        ranked_sent_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for sent_id in ranked_sent_ids:
            if sent_id not in selected_ids:
                selected_ids.append(sent_id)
            if len(selected_ids) >= max_sent_num:
                break

    evidence_sentences = [all_sentences[sent_id] for sent_id in selected_ids]
    return selected_ids, evidence_sentences


def normalize_retrieval_score(score: float) -> float:
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def compute_relation_score(
    retrieval_score: float,
    summary_score: float,
    evidence_score: float,
    type_score: float,
    distance_score: float,
) -> float:
    return (
        RETRIEVAL_WEIGHT * retrieval_score
        + SUMMARY_WEIGHT * summary_score
        + EVIDENCE_WEIGHT * evidence_score
        + TYPE_WEIGHT * type_score
        + DISTANCE_WEIGHT * distance_score
    )


def parse_retrieval_row(row: list[Any]) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    if not isinstance(row, list) or len(row) != 2:
        raise ValueError("Unexpected retrieval row format.")
    base_info = row[0]
    candidate_rels = row[1]
    return base_info, candidate_rels


def decide_routing(candidate_relations: list[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    if not candidate_relations:
        return "discard", [], []

    if len(candidate_relations) == 1:
        top1 = candidate_relations[0]
        if top1["final_score"] >= DIRECT_VERIFY_THRESHOLD:
            return "direct_verify", [top1["rel"]], []
        return "multiple_choice", [], [top1["rel"]]

    top1 = candidate_relations[0]
    top2 = candidate_relations[1]
    if top1["final_score"] >= DIRECT_VERIFY_THRESHOLD and (
        top1["final_score"] - top2["final_score"] >= DIRECT_VERIFY_GAP
    ):
        return "direct_verify", [top1["rel"]], []

    return "multiple_choice", [], [item["rel"] for item in candidate_relations[:TOPM_RERANK]]


data_name = os.getenv("DATA_NAME", "dev")
doc_name = "docred"
doc_dir = f"../data/{doc_name}/"
doc_filename = f"{doc_dir}{data_name}.json"
docred_df = read_json_file_as_df(doc_filename)
docred_len = len(docred_df)
doc_start, doc_end = get_doc_range(docred_len)
range_tag = get_range_tag(doc_start, doc_end)

retrieval_file_path = f"../data/retrieval_from_train/{data_name}/path-k20-{doc_name}_{range_tag}.jsonl"
relation_summary_file_path = (
    f"../data/check_result_relation_summary_jsonl/{data_name}/"
    f"result_{doc_name}_{data_name}_relation_summary_{range_tag}.jsonl"
)

retrieval_rows = read_jsonl(retrieval_file_path)
relation_summary_rows = read_jsonl(relation_summary_file_path)

if len(retrieval_rows) != len(relation_summary_rows):
    raise ValueError("Retrieval rows and relation summary rows are misaligned.")

rel_info = eval(open(f"../data/{doc_name}/rel_info.json", "r", encoding="utf-8").read())
reverse_rel_info = {v: k for k, v in rel_info.items()}
rel_description_dict = json.load(open("../relation_description.json", "r", encoding="utf-8"))
rel_judge_dict = build_rel_judge_dict("../rel_judge.txt")

model = SentenceTransformer("../data/all-mpnet-base")
relation_description_embeddings = {
    rel: model.encode(description, convert_to_tensor=True)
    for rel, description in rel_description_dict.items()
}

save_rows: list[dict[str, Any]] = []

print("data len:", len(retrieval_rows))
print("----------------------------------")

for idx, retrieval_row in enumerate(retrieval_rows):
    print(idx)
    base_info, candidate_rels = parse_retrieval_row(retrieval_row)

    title = base_info["title"]
    entity_h = base_info["entity_h"]
    entity_t = base_info["entity_t"]
    entity_h_id = base_info["entity_h_id"]
    entity_t_id = base_info["entity_t_id"]
    entities_description = base_info["entities_description"]
    doc_id = get_docid(title, docred_df)

    evidence_sentence_ids, evidence_sentences = select_evidence_sentences(
        doc_id=doc_id,
        entity_h_id=entity_h_id,
        entity_t_id=entity_t_id,
        entities_description=entities_description,
        df=docred_df,
        model=model,
        max_sent_num=EVIDENCE_SENT_NUM,
    )

    evidence_text = " ".join(evidence_sentences).strip()
    summary_embedding = model.encode(entities_description, convert_to_tensor=True)
    evidence_embedding = model.encode(evidence_text, convert_to_tensor=True) if evidence_text else None
    distance_score = compute_distance_score(doc_id, entity_h_id, entity_t_id, docred_df)

    reranked_candidates: list[dict[str, Any]] = []

    for rel, rel_meta in candidate_rels[:TOPK_RETRIEVAL]:
        if rel == "no_relation":
            continue

        relation_embedding = relation_description_embeddings[rel]
        retrieval_score = normalize_retrieval_score(rel_meta["score"])
        summary_score = float(util.cos_sim(summary_embedding, relation_embedding).item())
        summary_score = max(0.0, min(1.0, (summary_score + 1.0) / 2.0))

        if evidence_embedding is None:
            evidence_score = 0.0
        else:
            evidence_score = float(util.cos_sim(evidence_embedding, relation_embedding).item())
            evidence_score = max(0.0, min(1.0, (evidence_score + 1.0) / 2.0))

        type_score = 1.0
        if USE_TYPE_CONSTRAINT:
            type_score = 1.0 if rel_h_t_judge(
                rel,
                entity_h_id,
                entity_t_id,
                rel_judge_dict,
                reverse_rel_info,
                doc_id,
                docred_df,
            ) else 0.0

        final_score = compute_relation_score(
            retrieval_score=retrieval_score,
            summary_score=summary_score,
            evidence_score=evidence_score,
            type_score=type_score,
            distance_score=distance_score,
        )

        reranked_candidates.append(
            {
                "rel": rel,
                "final_score": round(final_score, 6),
                "retrieval_score": round(retrieval_score, 6),
                "summary_score": round(summary_score, 6),
                "evidence_score": round(evidence_score, 6),
                "type_score": round(type_score, 6),
                "distance_score": round(distance_score, 6),
                "train_title": rel_meta["train_title"],
                "train_entity_h": rel_meta["entity_h"],
                "train_entity_t": rel_meta["entity_t"],
                "train_entity_h_id": rel_meta["entity_h_id"],
                "train_entity_t_id": rel_meta["entity_t_id"],
            }
        )

    reranked_candidates.sort(key=lambda item: item["final_score"], reverse=True)
    top_candidates = reranked_candidates[:TOPM_RERANK]
    routing_decision, direct_relations, mc_relations = decide_routing(top_candidates)

    save_rows.append(
        {
            "title": title,
            "doc_id": doc_id,
            "entity_h": entity_h,
            "entity_t": entity_t,
            "entity_h_id": entity_h_id,
            "entity_t_id": entity_t_id,
            "entities_description": entities_description,
            "evidence_sentence_ids": evidence_sentence_ids,
            "evidence_sentences": evidence_sentences,
            "candidate_relations": reranked_candidates,
            "top_relations": [item["rel"] for item in top_candidates],
            "routing_decision": routing_decision,
            "direct_relations": direct_relations,
            "multiple_choice_relations": mc_relations,
        }
    )

save_name = f"../data/retrieval_rerank/{data_name}/rerank-k20-{doc_name}_{range_tag}.jsonl"
save_to_jsonl(save_rows, save_name)
print(f"The result is saved in the file {save_name}")
