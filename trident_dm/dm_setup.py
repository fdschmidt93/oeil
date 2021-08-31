# train data
# val data
# other data -> predictions
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

# from src.utils import (filter_labels, filter_paths, get_cod, get_logger,
#                        get_summary_paths, read_file, read_label,
#                        tokenize_to_vec)


def get_cod(path: Union[str, Path]) -> str:
    """Get COD from a filepath string."""
    return "/".join(str(path).split("/")[-4:-2])


def read_file(path: Path) -> List[str]:
    with open(path, "r") as file:
        lines = [line.strip() for line in file]
    return lines


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


class PlainDataset(Dataset):
    def __init__(self, X: list[str], y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        assert self.y.shape[0] == len(self.X)
        return len(self.X)

    def __getitem__(self, index) -> tuple[str, torch.Tensor]:
        return {"text": self.X[index], "labels": self.y[index]}


from pathlib import Path

home = str(Path.home())
data_dir = f"{home}/eia/oeil/data/"


def setup(self, *args, **kwargs):
    # data = Path(config.data_dir)
    #   read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
    labels, y = read_label(f"{data_dir}/labels/all.csv")
    #   get all pathlib.Paths to legislative proposal summaries

    lp_paths = get_summary_paths(f"{data_dir}/summaries/raw", final_act=False)
    #   split Paths into (labelled, unlabelled) paths
    lp_train_paths, lp_other_paths = filter_labels(lp_paths, labels)

    #   Loading and tf-idf weighting data
    fa_paths = get_summary_paths(f"{data_dir}/summaries/raw", final_act=True)
    #   split Paths into (labelled, unlabelled) paths
    fa_train_paths, fa_other_paths = filter_labels(fa_paths, labels)

    fa_sum = [read_file(x) for x in fa_paths]
    fa_cod = [get_cod(p) for p in fa_paths]

    lp_train_sum = [read_file(x)[0] for x in lp_train_paths]
    lp_other_sum = [read_file(x)[0] for x in lp_other_paths]
    self.data_train = PlainDataset(X=lp_train_sum, y=torch.from_numpy(y))

    lp_validation_data = pd.read_csv(f"{data_dir}/lp_validation_data.csv")
    fa_validation_data = pd.read_csv(f"{data_dir}/fa_validation_data.csv")

    lp_validation_paths, _ = filter_labels(
        lp_paths, lp_validation_data.loc[:, "COD"].values.tolist()
    )
    fa_validation_paths, _ = filter_labels(
        fa_paths, fa_validation_data.loc[:, "COD"].values.tolist()
    )

    lp_validation_sum = [read_file(x)[0] for x in lp_validation_paths]
    fa_validation_sum = [read_file(x)[0] for x in fa_validation_paths]
    test_text = lp_validation_sum + fa_validation_sum

    lp_test_labels = torch.from_numpy(lp_validation_data.loc[:, "LP_Coder_1"].values)
    fa_test_labels = torch.from_numpy(fa_validation_data.loc[:, "FA_Coder_1"].values)
    test_labels = torch.cat([lp_test_labels, fa_test_labels])
    self.data_test = PlainDataset(X=test_text, y=test_labels)
    self.data_val = self.data_test
