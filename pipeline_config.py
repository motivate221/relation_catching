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


def read_json_file_as_df(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as fr:
        return pd.read_json(StringIO(fr.read()))
