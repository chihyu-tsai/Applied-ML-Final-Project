import torch
from typing import Any, List, Tuple
import pandas as pd
import re
from urllib.parse import urlparse

from pathlib import Path

CKPT_PATH = Path(__file__).resolve().parent / "model.pt"
_ckpt = torch.load(CKPT_PATH, map_location="cpu")
stoi = _ckpt["stoi"]
PAD_id = _ckpt["PAD_id"]
UNK_id = _ckpt["UNK_id"]
MAX_LEN = _ckpt["MAX_LEN"]


label_map = {"fox": 0, "nbc": 1}

# helper functions
def tokenize(line: str):
    clean_lines = line.lower().strip()
    clean_lines = re.sub(r"[^0-9a-z\s]", " ", clean_lines)
    clean_lines = re.sub(r"\s+", " ", clean_lines).strip()
    tokens = clean_lines.split(" ")
    return tokens


def encode(lines, max_length):
    vocabs = []
    for line in lines:
        tokens = tokenize(line)
        row = [stoi.get(token, UNK_id) for token in tokens][:max_length]
        row += [PAD_id] * (max_length - len(row))
        vocabs.append(row)
    return torch.tensor(vocabs, dtype = torch.long)


# identify the source from ulr
def obtain_source(url: str) -> str:
    netloc = urlparse(url).netloc
    if netloc.endswith("foxnews.com"):
        return "fox"
    elif netloc.endswith("nbcnews.com"):
        return "nbc"
    else:
        return "unknown"

# get the title from url
def get_title(url: str) -> str:
    try:
        path = urlparse(url).path
        parts = [p for p in path.strip().split("/") if p]
        if not parts:
            return ""

        last = parts[-1]
        # strip the extension
        if "." in last:
            last = last.split(".")[0]

        # special case for nbc
        chunks = last.split("-")
        if chunks and chunks[-1] and chunks[-1][-1].isdigit():
            chunks = chunks[:-1] or chunks

        title = " ".join(chunks)
        return title

    except Exception:
        raise ValueError(f"Invalid URL: {url}")


def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    # read the csv file
    df = pd.read_csv(path)

    with_label_ds = df.assign(source=df['url'].apply(obtain_source))
    with_label_ds = with_label_ds.assign(
        label=with_label_ds["source"].map(label_map),
        headline=with_label_ds["url"].apply(get_title),
    )
    # encode the headlines
    X = encode(with_label_ds['headline'].tolist(), MAX_LEN)
    y = torch.tensor(with_label_ds['source'].map(label_map).values, dtype=torch.long)

    return X, y

    # raise NotImplementedError("Implement prepare_data(csv_path) -> (X, y).")

