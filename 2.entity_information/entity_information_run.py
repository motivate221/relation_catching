import json
import os
import sys
import glob

import requests

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import get_doc_range, get_range_tag, read_json_file_as_df


def save_to_jsonl(data, jsonl_file):
    output_dir = os.path.dirname(jsonl_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile)
            jsonlfile.write('\n')


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def count_jsonl_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def get_existing_shards(data_name, doc_name, range_tag):
    pattern = f"../data/entity_information_run/{data_name}/result_{doc_name}_{data_name}_entity_information_{range_tag}_*.jsonl"
    shard_infos = []
    for file_path in sorted(glob.glob(pattern)):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        suffix = base_name.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            continue
        shard_infos.append({
            "path": file_path,
            "save_id": int(suffix),
            "line_count": count_jsonl_lines(file_path),
        })
    shard_infos.sort(key=lambda item: item["save_id"])
    return shard_infos


def extract_assistant_response(response):
    assistant_start_pos = response.find("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_start_pos == -1:
        return response.strip()

    result_response_1 = response[assistant_start_pos:]
    assistant_end_pos = result_response_1.find("<|eot_id|>")
    if assistant_end_pos == -1:
        result_response_2 = result_response_1
    else:
        result_response_2 = result_response_1[:assistant_end_pos]

    result_response = result_response_2.replace("<|start_header_id|>assistant<|end_header_id|>", "", 1)
    return result_response.strip()


def request_model(prompt_list, temperature, max_new_tokens):
    data = {
        "system_prompt": "",
        "message": prompt_list,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }

    try:
        response = requests.post("http://127.0.0.1:6006", json=data, timeout=600)
        response.raise_for_status()
        result_list = response.json()
        if not isinstance(result_list, list):
            return []
        return result_list
    except requests.RequestException as e:
        print(f"request failed: {type(e).__name__}: {e}")
        return []
    except ValueError as e:
        print(f"invalid json response: {type(e).__name__}: {e}")
        return []


def run_one(prompt):
    result_list = request_model([prompt], temperature=0.9, max_new_tokens=50)
    if not result_list:
        return ""
    return extract_assistant_response(result_list[0])


def run_list(prompt_list):
    result_list = request_model(prompt_list, temperature=0.9, max_new_tokens=50)
    response_list_run = []

    if not result_list:
        for prompt in prompt_list:
            response_list_run.append(run_one(prompt))
    else:
        for response in result_list:
            response_list_run.append(extract_assistant_response(response))

    return response_list_run


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

len_data = len(jsonl_data)

print("data len: ", len_data)
print("----------------------------------")

batch_size = 20
prompt_list = []
response_list = []
id_list = []

existing_shards = get_existing_shards(data_name, doc_name, range_tag)
saved_line_count = sum(item["line_count"] for item in existing_shards)
start = min(saved_line_count, len_data)
save_id = (existing_shards[-1]["save_id"] + 1) if existing_shards else 0
end = len_data
save_data_list = []
save_cnt = 0

if existing_shards:
    print(f"resume detected: {len(existing_shards)} shard(s), {saved_line_count} record(s) already saved")
    print(f"resuming from id {start}, next save_id {save_id}")

for id in range(start, end):
    if len(prompt_list) == batch_size:
        response_list = run_list(prompt_list)

        for i in range(batch_size):
            now_id = id_list[i]
            if i >= len(response_list):
                jsonl_data[now_id]["response"] = []
                print(f"{now_id} data query failed")
            else:
                response = response_list[i]
                jsonl_data[now_id]["response"] = response
                print(f"{now_id} data query completed")

            save_data_list.append(jsonl_data[now_id])

        prompt_list.clear()
        id_list.clear()

    if save_cnt == 200:
        save_name = f"../data/entity_information_run/{data_name}/result_{doc_name}_{data_name}_entity_information_{range_tag}_{save_id}.jsonl"
        save_to_jsonl(save_data_list, save_name)
        print(f"The result is saved in the file {save_name}")
        save_id += 1
        save_cnt = 0
        save_data_list.clear()

    prompt = jsonl_data[id]["prompt"]
    prompt_list.append(prompt)
    id_list.append(id)
    save_cnt += 1


if len(prompt_list) > 0:
    response_list = run_list(prompt_list)

    for i in range(len(id_list)):
        response = response_list[i]
        now_id = id_list[i]
        jsonl_data[now_id]["response"] = response
        print(f"{now_id} data query completed")

        save_data_list.append(jsonl_data[now_id])

    prompt_list.clear()
    id_list.clear()

    save_name = f"../data/entity_information_run/{data_name}/result_{doc_name}_{data_name}_entity_information_{range_tag}_{save_id}.jsonl"
    save_to_jsonl(save_data_list, save_name)
    print(f"The result is saved in the file {save_name}")
    save_id += 1
    save_cnt = 0
    save_data_list.clear()
