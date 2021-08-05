import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pycm import ConfusionMatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from tokenizers import Tokenizer

from src.utils import (
    filter_labels,
    filter_paths,
    get_cod,
    get_summary_paths,
    read_file,
    read_label,
    tokenize_to_vec,
)
from src.weighting import tf_idf, z_norm

# Prepare file paths
CWD = Path.cwd()
DATA = Path("/work/fabiasch/eia/data")
SEG = DATA.joinpath("filtered2")
tokenizer = Tokenizer.from_file("./tokenizer.lr.json")
id2token = {
    v: k for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
}
tokens = list(id2token.values())

# Read files
# read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
# labels, y = read_label("./label.csv")
# get all pathlib.Paths to legislative proposal summaries
final_act_paths = get_summary_paths(SEG, final_act=True)
final_act_sum = [read_file(x) for x in final_act_paths]
# load pre-trained huggingface tokenizer
tokenizer = Tokenizer.from_file("./tokenizer.lr.json")


# get all pathlib.Paths to legislative proposal summaries
legis_sum_paths = get_summary_paths(SEG, final_act=False)
# split Paths into (label, other) paths
# load pre-trained huggingface tokenizer

# Prepare training data
# read sentence-splitted legislative proposal summaries into List[List[str]]
legis_sum = [read_file(x) for x in legis_sum_paths]
# stack document-token count arrays
X_train = np.stack([tokenize_to_vec(doc, tokenizer) for doc in legis_sum])
# bullet_cod = dict(zip(list(labels.keys()), X[:, 98].tolist()))
# tf-idf transformation
X_train, idf = tf_idf(X_train, return_idf=True)

X_final_act = np.stack([tokenize_to_vec(doc, tokenizer) for doc in final_act_sum])
cod = [get_cod(p) for p in final_act_paths]
token_counts = pd.DataFrame(index=cod, data=X_final_act, columns=tokens)
token_counts.to_csv("final_act_token_counts.csv")

# X_final_act = tf_idf(X_final_act, idf=idf)
X_final_act = tf_idf(X_final_act)
token_counts = pd.DataFrame(index=cod, data=X_final_act, columns=tokens)
token_counts.to_csv("final_act_tf_idf_fa-idf.csv")

# load model
model_path = "./lr.model.bin"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# predict
proposal_preds = model.predict(X_train)
proposal_probs = np.around(model.predict_proba(X_train)[:, 1], 2)
final_act_preds = model.predict(X_final_act)
final_act_probs = np.around(model.predict_proba(X_final_act)[:, 1], 2)

legis_sum_cod = [get_cod(p) for p in legis_sum_paths]
final_act_cod = [get_cod(p) for p in final_act_paths]
overlapping_paths = list(set(legis_sum_cod).intersection(set(final_act_cod)))

# filtering
filtered_proposal_preds = filter_paths(
    proposal_preds, legis_sum_paths, overlapping_paths
)
filtered_proposal_probs = np.array(
    filter_paths(proposal_probs, legis_sum_paths, overlapping_paths)
)
filtered_final_act_preds = filter_paths(
    final_act_preds, final_act_paths, overlapping_paths
)
filtered_final_act_probs = np.array(
    filter_paths(final_act_probs, final_act_paths, overlapping_paths)
)


df_dico = {
    "COD": overlapping_paths,
    "LR_Proposal_Prediction": filtered_proposal_preds,
    "LR_Proposal_Probability": filtered_proposal_probs,
    "LR_Final_Act_Prediction": filtered_final_act_preds,
    "LR_Final_Act_Probability": filtered_final_act_probs,
    "LR_Same_Class": np.array(filtered_proposal_preds)
    == np.array(filtered_final_act_preds),
    "LR_Prob_dif": np.around(filtered_final_act_probs - filtered_proposal_probs, 2),
}
df = pd.DataFrame.from_dict(df_dico)
df = df.sort_values("COD", axis=0)
df.to_csv("210802_lr.full-preds_fa-idf-reweighted.csv", index=False)

old = pd.read_csv("./210802_lr_proposal-to-act.csv")
old = old.sort_values("COD", axis=0)
df["LR_Probability_v210531-v210514"] = np.around(
    df["LR_Proposal_Probability"].values - old["LR_Proposal_Probability"].values, 2
)
df.to_csv("210531_lr.full-preds.csv", index=False)

diff_dico = {c: np.around(d, 2) for c, d in zip(cod, diff) if np.abs(d) > 0.1}


# TODO
# 2) match legislative acts
# 3) write to df
# sort :X
