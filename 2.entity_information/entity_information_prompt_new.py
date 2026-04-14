import pandas as pd
import json
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline_config import get_doc_range, get_range_tag, read_json_file_as_df


def save_to_jsonl(data, jsonl_file):
    output_dir = os.path.dirname(jsonl_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for item in data:
            json.dump(item, jsonlfile, ensure_ascii=False)
            jsonlfile.write('\n')

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        fallback_text = str(text).encode("gbk", errors="replace").decode("gbk", errors="replace")
        print(fallback_text)

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

def get_prompt_entity(title, doc, entity):
    prompt = f"""The text is as follows:

{title}
{doc}

What is "{entity}"? (For example, US is a country and 12 is a number) Answer in one sentence.Only output answers without outputting anything else.
The answer is:"""
    return prompt

def get_doc_entitys(doc_id, df):
    entity_list = []
    for entity in df['vertexSet'][doc_id]:
        name_list = []
        for entity_name in entity:
            name = entity_name['name']
            name_list.append(name)

        unique_name_list = list(set(name_list))
        entity_list.append(unique_name_list)

    return entity_list

def get_doc(doc_id, df):
    sentence_str = ""
    for sentence in df['sents'][doc_id]:
        for word in sentence:
            sentence_str += word
            sentence_str += " "
    return sentence_str

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:

            data.append(json.loads(line))
    return data

data_name = os.getenv("DATA_NAME", "dev")

doc_name = "docred"
doc_dir = f'../data/{doc_name}/'
doc_filename = f"{doc_dir}{data_name}.json"
docred_df = read_json_file_as_df(doc_filename)
docred_len = len(docred_df)
doc_start, doc_end = get_doc_range(docred_len)
range_tag = get_range_tag(doc_start, doc_end)

ok_entitys_dict = {}

if data_name == "train_annotated" or data_name == "train":
    for doc_id in range(docred_len):
        label_list = docred_df['labels'][doc_id]
        title = docred_df['title'][doc_id]
        for label in label_list:
            entity_h_id = label['h']
            entity_t_id = label['t']

            if title not in ok_entitys_dict:
                ok_entitys_dict[title] = {}
            if entity_h_id not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_h_id] = ""

            if entity_t_id not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_t_id] = ""

            ok_entitys_dict[title][entity_h_id] = "ok"
            ok_entitys_dict[title][entity_t_id] = "ok"

else:
    file_pattern = f"../data/get_entity_pair_selection_label/{data_name}/{doc_name}_{data_name}_entity_pair_selection_{range_tag}_answer-*.jsonl"
    for file_path in sorted(glob.glob(file_pattern)):
        for data in read_jsonl(file_path):
            entity_h_id = data["h_idx"]
            entity_t_id = data["t_idx"]
            title = data["title"]

            if title not in ok_entitys_dict:
                ok_entitys_dict[title] = {}
            if entity_h_id not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_h_id] = ""

            if entity_t_id not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_t_id] = ""

            ok_entitys_dict[title][entity_h_id] = "ok"
            ok_entitys_dict[title][entity_t_id] = "ok"


start, end = doc_start, doc_end


final_list = []
for doc_id in range(start, end):

    title = get_doc_title(doc_id, docred_df)
    doc = get_doc(doc_id, docred_df)
    entitys_list = get_doc_entitys(doc_id, docred_df)

    for index_1 in range(len(entitys_list)):
        for entity_name in entitys_list[index_1]:

            entity_id = get_entity_id(entity_name, docred_df, doc_id)

            if title not in ok_entitys_dict:
                ok_entitys_dict[title] = {}
            if entity_id not in ok_entitys_dict[title]:
                ok_entitys_dict[title][entity_id] = ""

            if ok_entitys_dict[title][entity_id] != "ok":
                continue

            prompt = get_prompt_entity(title, doc, entity_name)


            data_dict = {}
            data_dict["title"] = title
            data_dict["entity"] = entity_name
            data_dict["entity_id"] = entity_id
            data_dict["prompt"] = prompt
            data_dict["response"] = ""
            final_list.append(data_dict)

    safe_print(f"Doc:{title} prompt over")


jsonl_file = f"../data/entity_information_prompt/{data_name}/prompt_{doc_name}_{data_name}_entity_information_doc{range_tag}.jsonl"

save_to_jsonl(final_list, jsonl_file)
print(f"The result is saved in the file {jsonl_file}")
