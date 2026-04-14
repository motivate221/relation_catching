import json
import glob
import os
import re
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import get_doc_range, get_range_tag, read_json_file_as_df

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


def list_shard_files(pattern):
    shard_files = glob.glob(pattern)
    shard_files.sort(
        key=lambda file_path: int(re.search(r'_(\d+)\.jsonl$', os.path.basename(file_path)).group(1))
        if re.search(r'_(\d+)\.jsonl$', os.path.basename(file_path))
        else 10**9
    )
    if not shard_files:
        raise FileNotFoundError(f"No shard file matched pattern: {pattern}")
    return shard_files


data_name = os.getenv("DATA_NAME", "dev")

doc_name = "docred"
doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}.json"
docred_df = read_json_file_as_df(doc_filename)
docred_len = len(docred_df)
doc_start, doc_end = get_doc_range(docred_len)
range_tag = get_range_tag(doc_start, doc_end)


file_path = f"../data/entity_information_prompt/{data_name}/prompt_{doc_name}_{data_name}_entity_information_doc{range_tag}.jsonl"

jsonl_data = read_jsonl(file_path)

cnt = 0
shard_pattern = f"../data/entity_information_run/{data_name}/result_{doc_name}_{data_name}_entity_information_{range_tag}_*.jsonl"
shard_files = list_shard_files(shard_pattern)
print("all doc number:", len(shard_files))

save_list = []

for jsonl_file_path in shard_files:
    jsonl_data = read_jsonl(jsonl_file_path)

    for item in jsonl_data:
        if item['response'] == "":
            cnt += 1
            print("-------------------There is an empty response------------------")
        else:
            data_dict = {}
            data_dict["title"] = item["title"]
            data_dict["entity"] = item["entity"]
            data_dict["entity_id"] = item["entity_id"]
            data_dict["prompt"] = item["prompt"]
            data_dict["response"] = item["response"]
            save_list.append(data_dict)


save_path = f"../data/entity_information/{data_name}/result_{doc_name}_{data_name}_entity_information_{range_tag}.jsonl"
save_to_jsonl(save_list, save_path)
print(f"The result is saved in the file {save_path}")
print(f"There are {cnt} empty data records")
