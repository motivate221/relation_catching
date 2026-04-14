import json
import torch
import os
import sys
import glob
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import get_doc_range, get_range_tag, read_json_file_as_df


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(arg) for arg in args)
        safe_text = text.encode("gbk", errors="ignore").decode("gbk", errors="ignore")
        print(safe_text, **kwargs)

def cos(emb1,emb2):
    return util.pytorch_cos_sim(emb1, emb2)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_to_jsonl(data, jsonl_file):
    output_dir = os.path.dirname(jsonl_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')


def extract_range_tag(file_path):
    file_name = os.path.basename(file_path)
    match = re.search(r'_(\d+-\d+)\.(jsonl|npy)$', file_name)
    if not match:
        return None
    return match.group(1)


def parse_range_tag(range_tag):
    start_str, end_str = range_tag.split('-')
    start = int(start_str)
    end = int(end_str)
    return start, end


def choose_largest_common_cache_pair(jsonl_pattern, npy_pattern):
    jsonl_files = sorted(glob.glob(jsonl_pattern))
    npy_files = sorted(glob.glob(npy_pattern))

    if not jsonl_files:
        raise FileNotFoundError(f"No file matched pattern: {jsonl_pattern}")
    if not npy_files:
        raise FileNotFoundError(f"No file matched pattern: {npy_pattern}")

    jsonl_by_tag = {}
    for file_path in jsonl_files:
        range_tag = extract_range_tag(file_path)
        if range_tag is not None:
            jsonl_by_tag[range_tag] = file_path

    npy_by_tag = {}
    for file_path in npy_files:
        range_tag = extract_range_tag(file_path)
        if range_tag is not None:
            npy_by_tag[range_tag] = file_path

    common_range_tags = sorted(set(jsonl_by_tag.keys()) & set(npy_by_tag.keys()))
    if not common_range_tags:
        raise FileNotFoundError(
            "No common range tag found between train relation-summary cache and train embeddings cache."
        )

    def sort_key(range_tag):
        start, end = parse_range_tag(range_tag)
        span = end - start
        return (span, end, -start)

    best_range_tag = max(common_range_tags, key=sort_key)
    return jsonl_by_tag[best_range_tag], npy_by_tag[best_range_tag], best_range_tag

def get_docid(title, df):
    for doc_id in range(len(df)):
        if title == df['title'][doc_id]:
            return doc_id
def judge_example(reverse_rel_info, tain_docred_df, title, rel, train_entity_h_id, train_entity_t_id):
    train_doc_id = get_docid(title, tain_docred_df)
    label_list = train_docred_df['labels'][train_doc_id]
    rel_id = reverse_rel_info[rel]
    flag = 0
    for label in label_list:
        if label['h'] == train_entity_h_id and label['t'] == train_entity_t_id and label['r'] == rel_id:
            flag = 1
    if flag == 1:
        return True

    return False


data_name = os.getenv("DATA_NAME", "dev")

doc_name = "docred"
doc_dir = f'../data/{doc_name}/'
dev_doc_filename = f"{doc_dir}{data_name}.json"
dev_docred_df = read_json_file_as_df(dev_doc_filename)
dev_docred_len = len(dev_docred_df)
doc_start, doc_end = get_doc_range(dev_docred_len)
range_tag = get_range_tag(doc_start, doc_end)

train_doc_filename = f"../data/{doc_name}/train_annotated.json"
train_docred_df = read_json_file_as_df(train_doc_filename)


info_fr = open(f'../data/{doc_name}/rel_info.json', 'r', encoding='utf-8')
rel_info = info_fr.read()
rel_info = eval(rel_info)

reverse_rel_info = {v: k for k, v in rel_info.items()}


train_file_path, train_embeddings_file_path, train_cache_range_tag = choose_largest_common_cache_pair(
    f"../data/check_result_relation_summary_jsonl/train_annotated/result_{doc_name}_train_annotated_relation_summary_*.jsonl",
    f"../data/get_embeddings/{doc_name}_train_annotated_embeddings_*.npy"
)
train_annotated_jsonl_data = read_jsonl(train_file_path)


dev_file_path = f"../data/check_result_relation_summary_jsonl/{data_name}/result_{doc_name}_{data_name}_relation_summary_{range_tag}.jsonl"
dev_jsonl_data = read_jsonl(dev_file_path)

safe_print("data loading successful")
safe_print('--------------------------------------')

train_embeddings = np.load(train_embeddings_file_path)
safe_print("train_embeddings loading successful")
safe_print('--------------------------------------')
safe_print(f"train cache range selected: {train_cache_range_tag}")
safe_print(f"train relation summary cache: {train_file_path}")
safe_print(f"train embeddings cache: {train_embeddings_file_path}")
safe_print('--------------------------------------')


dev_embeddings = np.load(f'../data/get_embeddings/{doc_name}_{data_name}_embeddings_{range_tag}.npy')
safe_print("dev_embeddings loading successful")
safe_print('--------------------------------------')

dev_len = len(dev_jsonl_data)

k = 20

data_sort_n = k
entity_sort_n = k
threshold = 0.0

dev_top_all = {}

safe_print("data len:", dev_len)


for id in range(dev_len):
    safe_print(id)
    query_sentence = dev_jsonl_data[id]['entities_description']
    entity_h_id = dev_jsonl_data[id]["entity_h_id"]
    entity_t_id = dev_jsonl_data[id]["entity_t_id"]
    dev_title = dev_jsonl_data[id]['title']

    if dev_title not in dev_top_all:
        dev_top_all[dev_title] = {}

    if entity_h_id not in dev_top_all[dev_title]:
        dev_top_all[dev_title][entity_h_id] = {}
    if entity_t_id not in dev_top_all[dev_title][entity_h_id]:
        dev_top_all[dev_title][entity_h_id][entity_t_id] = []

    if entity_t_id not in dev_top_all[dev_title]:
        dev_top_all[dev_title][entity_t_id] = {}
    if entity_h_id not in dev_top_all[dev_title][entity_t_id]:
        dev_top_all[dev_title][entity_t_id][entity_h_id] = []


    query_embedding = dev_embeddings[id]
    ress = cos(train_embeddings, dev_embeddings[id]).reshape(-1).tolist()

    sorted_indices = sorted(range(len(ress)), key=lambda i: ress[i], reverse=True)
    cnt = 0

    val_len = len(sorted_indices)
    for sort_id in range(val_len):
        train_id = sorted_indices[sort_id]
        if cnt >= data_sort_n:
            break

        if ress[train_id] < threshold:
            break

        for label in train_annotated_jsonl_data[train_id]['label_rel']:
            data_dict = {}
            data_dict['score'] = ress[train_id]
            data_dict['rel'] = label
            data_dict['train_title'] = train_annotated_jsonl_data[train_id]['title']
            if judge_example(reverse_rel_info, train_docred_df, train_annotated_jsonl_data[train_id]['title'], label, train_annotated_jsonl_data[train_id]['entity_h_id'], train_annotated_jsonl_data[train_id]['entity_t_id']):
                data_dict['entity_h'] = train_annotated_jsonl_data[train_id]['entity_h']
                data_dict['entity_t'] = train_annotated_jsonl_data[train_id]['entity_t']
                data_dict['entity_h_id'] = train_annotated_jsonl_data[train_id]['entity_h_id']
                data_dict['entity_t_id'] = train_annotated_jsonl_data[train_id]['entity_t_id']
            else:
                data_dict['entity_h'] = train_annotated_jsonl_data[train_id]['entity_t']
                data_dict['entity_t'] = train_annotated_jsonl_data[train_id]['entity_h']
                data_dict['entity_h_id'] = train_annotated_jsonl_data[train_id]['entity_t_id']
                data_dict['entity_t_id'] = train_annotated_jsonl_data[train_id]['entity_h_id']

            dev_top_all[dev_title][entity_h_id][entity_t_id].append(data_dict)
            dev_top_all[dev_title][entity_t_id][entity_h_id].append(data_dict)


        cnt += 1
    safe_print("-----------------------------")



sorted_dev_top = {}
for key, value in dev_top_all.items():
    sorted_dev_top[key] = {}
    for key_1, value_1 in value.items():
        sorted_dev_top[key][key_1] = {}
        for key_2, value_2 in value_1.items():
            sorted_dev_top[key][key_1][key_2] = []
            sorted_value = sorted(value_2, key=lambda x: x[list(x.keys())[0]], reverse=True)
            sorted_dev_top[key][key_1][key_2] = sorted_value

dev_result_rel = {}

for key, value in sorted_dev_top.items():

    dev_result_rel[key] = {}
    for key_1, value_1 in value.items():
        dev_result_rel[key][key_1] = {}
        for key_2, value_2 in value_1.items():
            dev_result_rel[key][key_1][key_2] = {}
            val_len = len(value_2)
            cnt = 0

            for val_id in range(val_len):

                if value_2[val_id]['rel'] != "no_relation":
                    if value_2[val_id]['rel'] not in dev_result_rel[key][key_1][key_2]:
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']] = {}
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['score'] = value_2[val_id]['score']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['train_title'] = value_2[val_id]['train_title']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_h'] = value_2[val_id]['entity_h']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_t'] = value_2[val_id]['entity_t']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_h_id'] = value_2[val_id]['entity_h_id']
                        dev_result_rel[key][key_1][key_2][value_2[val_id]['rel']]['entity_t_id'] = value_2[val_id]['entity_t_id']

                cnt += 1
                if cnt == entity_sort_n:
                    break


save_result_rel = {}


for key, value in dev_result_rel.items():
    save_result_rel[key] = {}
    for key_1, value_1 in value.items():
        save_result_rel[key][key_1] = {}
        for key_2, value_2 in value_1.items():
            save_result_rel[key][key_1][key_2] = []
            safe_print(f"doc:{key} entity_h:{key_1} entity_t:{key_2} relation number:{len(value_2)}")


            predict_rel_list = {}

            for key_3, value_3 in value_2.items():
                rel = key_3

                if rel not in predict_rel_list:
                    predict_rel_list[rel] = 1
                    save_result_rel[key][key_1][key_2].append((rel, value_3))




save_list = []

for data in dev_jsonl_data:
    entity_h_id = data["entity_h_id"]
    entity_t_id = data["entity_t_id"]
    dev_title = data['title']
    data_1 = data
    data_2 = []
    label_list = save_result_rel[dev_title][entity_h_id][entity_t_id]
    for label, value in label_list:
        data_2.append((label, value))
    save_data = []
    save_data.append(data_1)
    save_data.append(data_2)
    save_list.append(save_data)

save_name = f"../data/retrieval_from_train/{data_name}/path-k20-{doc_name}_{range_tag}.jsonl"
save_to_jsonl(save_list, save_name)
safe_print(f"The result is saved in the file {save_name}")
