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
)

from src.weighting import sif_weighting, z_norm, sif, average_cos_sim, min_max_norm
from typing import Tuple, List
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from sentence_transformers import SentenceTransformer
import pandas as pd

# Prepare file paths
CWD = Path.cwd()
DATA = Path("/work/fabiasch/eia/data")
SEG = DATA.joinpath("segmented")
EMB = "./data/emb/cc.en.300.vec"


# Read files
# read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
labels, y = read_label("./label.csv")
# get all pathlib.Paths to legislative proposal summaries
file_paths = get_summary_paths(SEG)
# split Paths into (label, other) paths
train_paths, other_paths = filter_labels(file_paths, labels)
# read sentence-segmented legislative proposal summaries into List[List[str]]
train_sum = [read_file(x) for x in train_paths]
other_sum = [read_file(x) for x in other_paths]

train_sum = [" ".join(x) for x in train_sum]
other_sum = [" ".join(x) for x in other_sum]


#

N_train = len(train_sum)

model_name = "stsb-distilbert-base"
path = CWD.joinpath(f'{model_name}_embeddings')
if not path.exists():
    device = "cuda:1"
    model = SentenceTransformer(model_name, device=device)
    train_embeds = model.encode(train_sum) 
    other_embeds = model.encode(other_sum) 
    embeds = np.vstack((train_embeds, other_embeds))
    np.save(str(path), embeds)
else:
    embeds = np.load(str(path)+'.npy')

# embed documents
X = z_norm(embeds)
X = embeds

# generate training data
X_train_docs = X[: N_train]
X_train, X_test, y_train, y_test = train_test_split(
    X_train_docs, y, test_size=0.2, random_state=42, stratify=y
)
# CV: exhaustive l2 parameter tuning
penalty = ["l2"]
C = np.logspace(0, 1, 100)
hparams = {"C": C, "penalty": penalty}
model = LogisticRegression(random_state=42, solver="liblinear", max_iter=20)
clf = GridSearchCV(model, hparams, cv=10, verbose=0, scoring='f1_macro')
clf.fit(X_train, y_train)
# best model
print(f"The best Logistic Regression model is: {clf.best_estimator_}")
print(f"The best CV validation score: {clf.best_score_}")

# hold out test set performance: 0.5
x_preds = clf.predict(X_test)
print((x_preds == y_test).mean())
print(classification_report(y_test, x_preds))
