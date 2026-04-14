import os

from fastapi import FastAPI, Request
import uvicorn, json, datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = "0"
CUDA_DEVICE = f"cuda:{DEVICE_ID}"
DEVICE_MAP = "auto" if DEVICE == "cuda" else None
MODEL_PATH = os.getenv("MODEL_PATH", "../model-path/")


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        fallback_text = str(text).encode("gbk", errors="replace").decode("gbk", errors="replace")
        print(fallback_text)


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def get_quantization_config():
    if DEVICE != "cuda":
        return None

    load_in_4bit = get_bool_env("MODEL_LOAD_IN_4BIT", True)
    load_in_8bit = get_bool_env("MODEL_LOAD_IN_8BIT", False)

    if load_in_4bit and load_in_8bit:
        raise ValueError("MODEL_LOAD_IN_4BIT and MODEL_LOAD_IN_8BIT cannot both be true.")

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


def load_model_and_tokenizer():
    quantization_config = get_quantization_config()
    model_kwargs = {}

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = DEVICE_MAP
    elif DEVICE == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = DEVICE_MAP
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if DEVICE == "cpu":
        model.to(torch.device("cpu"))

    model.eval()
    return model, tokenizer


app = FastAPI()



@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    system_prompt = json_post_list.get('system_prompt')
    message = json_post_list.get('message')
    temperature = json_post_list.get('temperature')
    max_new_tokens = json_post_list.get('max_new_tokens')


    inputs_list = []
    for prompt in message:
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs_list.append(inputs)



    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(inputs_list, return_tensors="pt", padding=True)
    if DEVICE == "cuda":
        inputs = inputs.to(torch.device(CUDA_DEVICE))

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    terminators = [tokenizer.eos_token_id]
    if eot_token_id is not None:
        terminators.append(eot_token_id)

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        attention_mask = inputs.attention_mask
    )

    response_list = []
    prompt_token_len = inputs.input_ids.shape[1]
    for seq in outputs:
        generated_seq = seq[prompt_token_len:]
        response = tokenizer.decode(generated_seq, skip_special_tokens=True)
        response_list.append(response.strip())

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    safe_print(time)
    safe_print(response_list)

    torch_gc()
    return response_list


if __name__ == '__main__':
    safe_print(f"MODEL_PATH={MODEL_PATH}")
    safe_print(f"DEVICE={DEVICE}")
    safe_print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
    safe_print(f"MODEL_LOAD_IN_4BIT={get_bool_env('MODEL_LOAD_IN_4BIT', True)}")
    safe_print(f"MODEL_LOAD_IN_8BIT={get_bool_env('MODEL_LOAD_IN_8BIT', False)}")

    model, tokenizer = load_model_and_tokenizer()
    uvicorn.run(app, host='127.0.0.1', port=6006, workers=1)
