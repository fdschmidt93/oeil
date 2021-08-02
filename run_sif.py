import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pycm import ConfusionMatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tokenizers import Tokenizer

from src.utils import (
    filter_labels,
    get_cod,
    get_summary_paths,
    get_tokenizer_emb,
    read_embeddings,
    read_file,
    read_label,
    tokenize,
    tokenize_to_vec,
)
from src.weighting import average_cos_sim, min_max_norm, sif, sif_weighting, z_norm

# Prepare file paths
CWD = Path.cwd()
DATA = Path("./data")
SEG = DATA.joinpath("preprocessed")
EMB = "../data/emb/cc.en.300.vec"


def main():
    # read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
    labels, y = read_label("./label_extra_pos.csv")
    # get all pathlib.Paths to legislative proposal summaries
    file_paths = get_summary_paths(SEG, final_act=False)
    # split Paths into (label, other) paths
    train_paths, other_paths = filter_labels(file_paths, labels)
    # read sentence-segmented legislative proposal summaries into List[List[str]]
    train_sum = [read_file(x) for x in train_paths]
    other_sum = [read_file(x) for x in other_paths]

    # load pre-trained huggingface tokenizer
    tokenizer = Tokenizer.from_file("./tokenizer.sif.json")

    word2id, emb = read_embeddings(EMB, n_tokens=60_000)
    emb = get_tokenizer_emb(tokenizer, word2id, emb)

    # Prepare training data
    # sif weighting: stack document-token count arrays
    train_docs = np.stack([tokenize_to_vec(doc, tokenizer) for doc in train_sum])
    other_docs = np.stack([tokenize_to_vec(doc, tokenizer) for doc in other_sum])
    docs = np.vstack([train_docs, other_docs]).astype(np.int32)
    emb = sif_weighting(docs.sum(0), emb)

    #
    train_docs = [tokenize(doc, tokenizer) for doc in train_sum]
    other_docs = [tokenize(doc, tokenizer) for doc in other_sum]

    # embed documents
    docs = train_docs + other_docs
    X, pc = sif(docs, emb, reduction="mean", remove_pc=1)
    X = z_norm(X)

    # generate training data
    X_train_docs = X[: len(train_docs)]
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_docs, y, test_size=0.2, random_state=42, stratify=y
    )
    # CV: exhaustive l2 parameter tuning
    penalty = ["l2"]
    C = np.logspace(0, 1, 100)
    hparams = {"C": C, "penalty": penalty}
    model = LogisticRegression(random_state=42, solver="liblinear", max_iter=20)
    clf = GridSearchCV(model, hparams, cv=10, verbose=0, scoring="f1_macro")
    clf.fit(X_train, y_train)

    # best model
    print(f"The best Logistic Regression model is: {clf.best_estimator_}")
    print(f"The best CV validation score: {clf.best_score_}")

    # hold out test set performance
    # x_preds = clf.predict(X_test)
    # print((x_preds == y_test).mean())
    # print(classification_report(y_test, x_preds))
    # ConfusionMatrix(actual_vector=y_test, predict_vector=x_preds).print_matrix()

    # best model
    print(clf.best_params_)
    print(f"Cross Validation f1_macro: {100*clf.best_score_:0.1f}%")
    x_preds = clf.predict(X_test)
    # print('Hold out subset accuracy: ', (x_preds == y_test).mean(), '%')
    print("Performance on held out subset from training data:")
    print(classification_report(y_test, x_preds))
    ConfusionMatrix(actual_vector=y_test, predict_vector=x_preds).print_matrix()

    clf.fit(X_train_docs, y)
    with open("sif.model.bin", "wb") as file:
        pickle.dump(clf, file)

    # LR
    X_test = X[len(train_docs) :]
    sif_lr_preds = clf.predict(X_test)
    sif_lr_probs = np.around(clf.predict_proba(X_test)[:, 1], 2)
    pos_sim = average_cos_sim(X_test, X_train_docs[y == 1])
    neg_sim = average_cos_sim(X_test, X_train_docs[y == 0])
    pos_sim_cos = np.around(pos_sim, 2)
    neg_sim_cos = np.around(pos_sim, 2)
    pos_sim_normed = np.around(min_max_norm(pos_sim), 2)
    neg_sim_normed = np.around(min_max_norm(neg_sim), 2)

    cod = [get_cod(p) for p in other_paths]
    out = {
        "COD": cod,
        "SIF_LR_Predictions": sif_lr_preds.tolist(),
        "SIF_LR_Probs": sif_lr_probs.tolist(),
        "SIF_Pos_Similarity": pos_sim_normed.tolist(),
        "SIF_Neg_Similarity": neg_sim_normed.tolist(),
    }
    df = pd.DataFrame.from_dict(out)
    df.to_csv("sif_output.csv")


if __name__ == "__main__":
    main()
