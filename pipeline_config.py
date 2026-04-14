from __future__ import annotations

import os
from io import StringIO

import pandas as pd


def get_doc_range(total_len: int) -> tuple[int, int]:
    start = int(os.getenv("DOC_START", "0"))
    end_env = os.getenv("DOC_END")
    end = total_len if end_env is None else int(end_env)

    start = max(0, min(start, total_len))
    end = max(start, min(end, total_len))
    return start, end


def get_range_tag(start: int, end: int) -> str:
    return f"{start}-{end}"


def get_use_rerank() -> bool:
    value = os.getenv("USE_RERANK", "true").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def get_method_tag() -> str:
    method_tag = os.getenv("METHOD_TAG", "").strip()
    if method_tag:
        return method_tag
    return "rerank" if get_use_rerank() else "baseline"


def append_method_tag(base_name: str, method_tag: str | None = None) -> str:
    tag = method_tag if method_tag is not None else get_method_tag()
    if not tag:
        return base_name
    return f"{base_name}_{tag}"


def read_json_file_as_df(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as fr:
        return pd.read_json(StringIO(fr.read()))
