import pickle
from typing import Dict
from tokenizers import Tokenizer
from src.utils import (
    read_file,
    read_label,
    get_summary_paths,
    tokenize,
    tokenize_to_vec,
    filter_labels,
    read_embeddings,
    get_tokenizer_emb,
    get_cod,
    filter_paths
)
from src.weighting import pc_removal, sif_weighting, z_norm, sif, average_cos_sim, min_max_norm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from pycm import ConfusionMatrix

# Prepare file paths
CWD = Path.cwd()
DATA = Path("/work/fabiasch/eia/data")
SEG = DATA.joinpath("filtered")
EMB = "../data/emb/cc.en.300.vec"
tokenizer = Tokenizer.from_file("./tokenizer.sif.json")

# Read files
# read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
# labels, y = read_label("./label.csv")
# get all pathlib.Paths to legislative proposal summaries
final_act_paths = get_summary_paths(SEG, final_act=True)
final_act_sum = [read_file(x) for x in final_act_paths]

# get all pathlib.Paths to legislative proposal summaries
labels, y = read_label("./label.csv")
# get all pathlib.Paths to legislative proposal summaries
file_paths = get_summary_paths(SEG, final_act=False)
# split Paths into (label, other) paths
train_paths, other_paths = filter_labels(file_paths, labels)
# read sentence-segmented legislative proposal summaries into List[List[str]]
train_sum = [read_file(x) for x in train_paths]
other_sum = [read_file(x) for x in other_paths]
legis_sum_paths = train_paths + other_paths
# split Paths into (label, other) paths
# load pre-trained huggingface tokenizer

word2id, emb = read_embeddings(EMB, n_tokens=60_000)
emb = get_tokenizer_emb(tokenizer, word2id, emb)
# Prepare training data
# read sentence-splitted legislative proposal summaries into List[List[str]]
train_docs = np.stack([tokenize_to_vec(doc, tokenizer) for doc in train_sum])
other_docs = np.stack([tokenize_to_vec(doc, tokenizer) for doc in other_sum])
docs = np.vstack([train_docs, other_docs]).astype(np.int32)
emb = sif_weighting(docs.sum(0), emb)

model_path = './sif.model.bin'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# embed documents
train_docs = [tokenize(doc, tokenizer) for doc in train_sum]
other_docs = [tokenize(doc, tokenizer) for doc in other_sum]
final_act_vec = np.stack([tokenize_to_vec(doc, tokenizer) for doc in
    final_act_sum]).astype(np.int32)

# embed documents
docs = train_docs + other_docs
X_legis_sum, pc = sif(docs, emb, reduction="mean", remove_pc=1)
X_legis_sum, stats = z_norm(X_legis_sum, return_stats=True)
X_train_docs = X_legis_sum[: len(train_docs)]

X_final_act, _ = sif(final_act_vec, emb, reduction='mean', pc=pc)
X_final_act = z_norm(X_final_act, stats=stats)

legis_sum_sif_lr_preds = model.predict(X_legis_sum)
legis_sum_sif_lr_probs = np.around(model.predict_proba(X_legis_sum)[:, 1], 2)
legis_sum_pos_sim = average_cos_sim(X_legis_sum, X_train_docs[y == 1])
legis_sum_neg_sim = average_cos_sim(X_legis_sum, X_train_docs[y == 0])
legis_sum_pos_sim = np.around(legis_sum_pos_sim, 2)
legis_sum_pos_sim = np.around(legis_sum_pos_sim, 2)
legis_sum_pos_sim_normed = np.around(min_max_norm(legis_sum_pos_sim), 2)
legis_sum_neg_sim_normed = np.around(min_max_norm(legis_sum_neg_sim), 2)


final_act_sif_lr_preds = model.predict(X_final_act)
final_act_sif_lr_probs = np.around(model.predict_proba(X_final_act)[:, 1], 2)
final_act_pos_sim = average_cos_sim(X_final_act, X_train_docs[y == 1])
final_act_neg_sim = average_cos_sim(X_final_act, X_train_docs[y == 0])
final_act_pos_sim = np.around(final_act_pos_sim, 2)
final_act_pos_sim = np.around(final_act_pos_sim, 2)
final_act_pos_sim_normed = np.around(min_max_norm(final_act_pos_sim), 2)
final_act_neg_sim_normed = np.around(min_max_norm(final_act_neg_sim), 2)

legis_sum_cod = [get_cod(p) for p in legis_sum_paths]
final_act_cod = [get_cod(p) for p in final_act_paths]
overlapping_paths = list(set(legis_sum_cod).intersection(set(final_act_cod)))

filtered_ls_sif_lr_preds = filter_paths(legis_sum_sif_lr_preds, legis_sum_paths, overlapping_paths)
filtered_ls_sif_lr_probs = filter_paths(legis_sum_sif_lr_probs, legis_sum_paths, overlapping_paths)
filtered_ls_neg_sim_normed = filter_paths(legis_sum_neg_sim_normed, legis_sum_paths, overlapping_paths)
filtered_ls_pos_sim_normed = filter_paths(legis_sum_pos_sim_normed, legis_sum_paths, overlapping_paths)

filtered_fa_sif_lr_preds = filter_paths(final_act_sif_lr_preds, final_act_paths, overlapping_paths)
filtered_fa_sif_lr_probs = filter_paths(final_act_sif_lr_probs, final_act_paths, overlapping_paths)
filtered_fa_neg_sim_normed = filter_paths(final_act_neg_sim_normed, final_act_paths, overlapping_paths)
filtered_fa_pos_sim_normed = filter_paths(final_act_pos_sim_normed, final_act_paths, overlapping_paths)

df_dico = {'COD': overlapping_paths,
            # 'Proposal_SIF_Pos_Similarity': filtered_ls_pos_sim_normed,
            # 'Proposal_SIF_Neg_Similarity': filtered_ls_neg_sim_normed,
            'Proposal_SIF_LR_Prediction': filtered_ls_sif_lr_preds,
            'Proposal_SIF_LR_Probability': filtered_ls_sif_lr_probs,
            # 'Final_Act_SIF_Pos_Similarity': filtered_fa_pos_sim_normed,
            # 'Final_Act_SIF_Neg_Similarity': filtered_fa_neg_sim_normed,
            'Final_Act_SIF_LR_Prediction': filtered_fa_sif_lr_preds,
            'Final_Act_SIF_LR_Probability': filtered_fa_sif_lr_probs,
            'SIF_Same_Class': filtered_ls_sif_lr_preds == filtered_fa_sif_lr_preds,
            'SIF_Prob_dif': np.around(np.array(filtered_fa_sif_lr_probs) - np.array(filtered_ls_sif_lr_probs), 2),
}
df = pd.DataFrame.from_dict(df_dico)

lr = pd.read_csv('./210514_lr.full-preds.csv')
out = pd.merge(lr, df, left_on='COD', right_on='COD')
df.to_csv('210514_sif.full-preds.csv', index=False)
out.to_csv('210514_full-scaling.csv', index=False)
