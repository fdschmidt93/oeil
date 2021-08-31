import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from tokenizers import Tokenizer

CWD = Path.cwd()
SEG = CWD.joinpath("segmented")
UNSEG = CWD.joinpath("unsegmented")
UNSEG_YEARS = UNSEG.glob("*")


def get_cod(path: Union[str, Path]) -> str:
    """Get COD from a filepath string."""
    return "/".join(str(path).split("/")[-4:-2])


def filter_paths(
    values: List[float], paths: List[Path], overlapping_paths: List[str]
) -> List[float]:
    out_val = []
    cod = [get_cod(p) for p in paths]
    for path in overlapping_paths:
        if path in cod:
            idx = cod.index(path)
            out_val.append(values[idx])
    return out_val


def read_embeddings(
    path: Union[str, Path], n_tokens: int = 200_000
) -> Tuple[Dict[str, int], np.ndarray]:
    word2id = {}
    embeddings = []
    with open(path, "r") as file:
        next(file)
        for i, line in enumerate(file):
            if i == n_tokens:
                break
            token, emb = line.rstrip().split(" ", maxsplit=1)
            if token not in word2id:
                word2id[token] = i
                embeddings.append(np.fromstring(emb, sep=" "))
    embeddings = np.stack(embeddings)
    return word2id, embeddings


def get_tokenizer_emb(
    tokenizer: Tokenizer, word2id: dict, embeddings: np.ndarray
) -> np.ndarray:
    vocab = {
        k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    }
    embs = [
        embeddings[word2id[token]]
        if token in word2id
        else np.zeros(embeddings.shape[-1])
        for token in vocab
    ]
    embs = np.stack(embs)
    return embs


def read_label(path: str, delimiter: str = ",") -> Tuple[Dict[str, int], np.ndarray]:
    """
    Reads the label from a csv-file into a dictionary of {COD: label} and np array of labels

    File example:

    Procedure,Category
    2010/0004(COD),-1
    2010/0258(COD),-1
    2014/0034(COD),-1
    2011/0411(COD),1
    2017/0220(COD),1
    """

    label = {}
    with open(path, "r") as file:
        # skip header
        next(file)
        for line in file:
            procedure, category = line.strip().split(delimiter)
            label[procedure] = 1 if int(category) == 1 else 0
    y = np.array(list(label.values()))
    return label, y


def filter_labels(
    file_paths: List[Path], labels: Union[List, Dict[str, int]]
) -> Tuple[List[Path], List[Path]]:
    """Separates file paths into file paths for labelled and unlabelled COD."""
    if isinstance(labels, dict):
        label_cod = list(labels.keys())
    else:
        label_cod = labels
    non_label_paths = [
        path for path in file_paths if not any(cod in str(path) for cod in label_cod)
    ]
    label_paths = [path for cod in label_cod for path in file_paths if cod in str(path)]
    assert len(labels) == len(label_paths)
    return label_paths, non_label_paths


def get_years(cwd: Path):
    return list([path for path in cwd.glob("*") if not path.is_file()])


def read_unsegmented_file(path: Path) -> Optional[str]:
    with open(path, "r") as file:
        for i, line in enumerate(file):
            try:
                text = line
                if not i:
                    return text
                else:
                    raise (ValueError(f"{path} caused problems."))
            except StopIteration:
                if not i:
                    return None
                else:
                    raise (ValueError(f"{path} caused problems."))


def read_file(path: Path) -> List[str]:
    with open(path, "r") as file:
        lines = [line.strip() for line in file]
    return lines


def tokenize(inputs: List[List[str]], tokenizer: Tokenizer) -> np.ndarray:
    tokens = np.array([y for x in tokenizer.encode_batch(inputs) for y in x.ids])
    return tokens[tokens > 0]


def tokenize_to_vec(document: List[str], tokenizer: Tokenizer, N: int):
    """Tokenize document, unpack tokens, and return counts."""
    vector = np.zeros(N, dtype=np.float32)
    encodings = tokenizer.encode_batch(document, add_special_tokens=True)
    ids, counts = np.unique(
        [id_ for x in encodings for id_ in x.ids], return_counts=True
    )
    try:
        counts[0] = 0
    except:
        import pudb

        pu.db
    vector[ids] = counts
    return vector


def get_summary_paths(folder: str, final_act: bool = False) -> List[Path]:
    # generate file paths
    folder = Path(folder)
    filetype = "legislative_proposal" if not final_act else "final_act"
    file_glob = "sum_" + filetype + "_*"
    files = []
    years = folder.glob("*")
    for year in years:
        codes = year.glob("*")
        for code in codes:
            # if str(code).split('/')[-1] == "0002(COD)":
            #     breakpoint()
            sum_path = code.joinpath("sum")
            proposals = list(sum_path.glob(file_glob))
            if proposals:
                arr = np.array([int(path.stem[-1]) for path in proposals])
                filepath = proposals[arr.argmax()]
                try:
                    if os.stat(str(filepath)).st_size == 0:
                        filepath = proposals[arr.argsort()[-2]]
                        if os.stat(str(filepath)).st_size == 0:
                            filepath = proposals[arr.argsort()[-3]]
                            if os.stat(str(filepath)).st_size == 0:
                                filepath = proposals[arr.argsort()[-4]]
                    files.append(filepath)
                except:
                    pass
    return files


def read_summaries(
    folder: Path, filter_cod: Optional[List[str]] = None,
):
    files = get_summary_paths(folder)

    # filter file paths
    if filter_cod is not None:
        files = [file for file in files if any(cod in str(file) for cod in filter_cod)]

    # read
    files = [read_file(path) for path in files]
    return files


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
