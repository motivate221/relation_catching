import json
import math
import os
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
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')



data_name = "dev"

doc_name = "docred"
doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}.json"
docred_df = read_json_file_as_df(doc_filename)
docred_len = len(docred_df)
doc_start, doc_end = get_doc_range(docred_len)
range_tag = get_range_tag(doc_start, doc_end)

file_path = f"../data/multiple_choice_prompt/{data_name}/multiple_choice_prompt-path-k20_{data_name}-{doc_name}_{range_tag}.jsonl"

save_doc_name = f"path-k20-{doc_name}_{range_tag}"

jsonl_data = read_jsonl(file_path)

len_data = len(jsonl_data)

cnt = 0
start = 0
end = math.ceil(len_data / 200)
print("all doc number:",end)

save_list = []

for jsonl_id in range(start, end):

    jsonl_file_path = f"../data/multiple_choice_run/{data_name}/{save_doc_name}/result_{doc_name}_{data_name}_multiple_choice_{save_doc_name}_{jsonl_id}.jsonl"
    jsonl_data = read_jsonl(jsonl_file_path)

    for item in jsonl_data:
        if item['response'] == "":
            cnt += 1
            print("-------------------There is an empty response------------------")
        else:
            data_dict = {}
            data_dict["title"] = item["title"]
            data_dict["doc_id"] = item["doc_id"]
            data_dict["prompt"] = item["prompt"]
            data_dict["prompt_rel"] = item["prompt_rel"]
            data_dict["entity_h"] = item["entity_h"]
            data_dict["entity_t"] = item["entity_t"]
            data_dict["entity_h_id"] = item["entity_h_id"]
            data_dict["entity_t_id"] = item["entity_t_id"]
            data_dict["response"] = item["response"]
            save_list.append(data_dict)

save_path = f"../data/check_result_multiple_choice_jsonl/{data_name}/result_{doc_name}_{data_name}_multiple_choice_{save_doc_name}.jsonl"
save_to_jsonl(save_list, save_path)
print(f"The result is saved in the file {save_path}")
print(f"There are {cnt} empty data records")
