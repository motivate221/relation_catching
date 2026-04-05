import pandas as pd
import numpy as np
import csv
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import get_doc_range, get_range_tag, read_json_file_as_df

def get_doc_title(doc_id, df):

    title = df['title'][doc_id]

    return title


def get_entity_id(entity, df, doc_id):
    len_doc = len(df['vertexSet'][doc_id])
    for entity_id in range(len_doc):
        for entity_name in df['vertexSet'][doc_id][entity_id]:
            if entity_name['name'] == entity:
                return entity_id

    return -1

def judge_rel(entity_h_id, entity_t_id, rel , doc_id, docred_df):

    for label in docred_df['labels'][doc_id]:
        if entity_h_id == label['h'] and entity_t_id == label['t'] and rel == label['r']:
            return True

    return False

def read_csv(csv_file):
    data_list = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_list.append(row)
    return data_list

def save_to_jsonl(data, jsonl_file):
    output_dir = os.path.dirname(jsonl_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii = False)
            jsonlfile.write('\n')
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_binary_answer(response):
    if not isinstance(response, str):
        return None
    match = re.search(r'\b(YES|NO)\b', response, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()



data_name = os.getenv("DATA_NAME", "dev")

doc_name = "docred"
doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}.json"
docred_df = read_json_file_as_df(doc_filename)
docred_len = len(docred_df)
doc_start, doc_end = get_doc_range(docred_len)
range_tag = get_range_tag(doc_start, doc_end)

info_fr = open(f'../data/{doc_name}/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)

reverse_rel_info = {v: k for k, v in rel_info.items()}

save_doc_name = f"k20-{doc_name}_{range_tag}"

file_path = f"../data/check_result_triplet_fact_judgement_jsonl/{data_name}/result_{doc_name}_{data_name}_triplet_fact_judgement_{save_doc_name}.jsonl"

jsonl_data = read_jsonl(file_path)


save_list = []

start = 0
length  = len(jsonl_data)

for id in range(start, length):

    data = jsonl_data[id]

    prompt_rel = data["prompt_rel"]
    prompt = data["prompt"]
    response = data["response"]
    entity_h_id = data["entity_h_id"]
    entity_t_id = data["entity_t_id"]
    doc_id = data["doc_id"]


    answer = extract_binary_answer(response)
    if answer == "YES":
        rel_dict = {}
        rel_dict["title"] = get_doc_title(doc_id, docred_df)
        rel_dict['h_idx'] = data["entity_h_id"]
        rel_dict['t_idx'] = data["entity_t_id"]
        rel_dict["r"] = reverse_rel_info[prompt_rel]
        save_list.append(rel_dict)

unique_save_list = [dict(t) for t in {tuple(d.items()) for d in save_list}]

save_name = f"../data/get_triplet_fact_judgement_label/{data_name}/{doc_name}_{data_name}_triplet_fact_judgement_{range_tag}_answer-{save_doc_name}.jsonl"
save_to_jsonl(unique_save_list, save_name)
print(f"The result is saved in the file {save_name}")
print("-----------------------------------------------")
