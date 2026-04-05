import pandas as pd
import numpy as np
import csv
import json
import os
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


data_name = "dev"

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
file_path = f"../data/get_triplet_fact_judgement_label/{data_name}/{doc_name}_{data_name}_triplet_fact_judgement_{range_tag}_answer-{save_doc_name}.jsonl"
jsonl_data = read_jsonl(file_path)

save_list = []

for data in jsonl_data:
    save_list.append(data)


with open(f'result/{doc_name}_{data_name}_ft_k20_{range_tag}.json', 'w') as json_file:
    json.dump(save_list, json_file)

print("Data has been saved as a JSON file.")
print("-----------------------------------------------")
