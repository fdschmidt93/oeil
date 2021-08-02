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
    get_cod,
    get_summary_paths,
    read_file,
    read_label,
    tokenize_to_vec,
)
from src.weighting import tf_idf, z_norm

CWD = Path.cwd()
DATA = Path("./data")
SEG = DATA.joinpath("preprocessed")
RANDOM_SEED = 38


def main():
    # read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
    labels, y = read_label("./label_extra_pos.csv")
    # get all pathlib.Paths to legislative proposal summaries
    file_paths = get_summary_paths(SEG, final_act=False)
    # split Paths into (label, other) paths
    train_paths, other_paths = filter_labels(file_paths, labels)
    # load pre-trained huggingface tokenizer
    tokenizer = Tokenizer.from_file("./tokenizer.lr.json")
    id2token = {
        v: k for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    }
    tokens = list(id2token.values())

    # Prepare training data
    # read sentence-splitted legislative proposal summaries into List[List[str]]
    train_sum = [read_file(x) for x in train_paths]
    other_sum = [read_file(x) for x in other_paths]
    all_sum = train_sum + other_sum
    # stack document-token count arrays
    X = np.stack([tokenize_to_vec(doc, tokenizer) for doc in all_sum])

    # get tokens
    # get counts
    # get cod
    cod = [get_cod(p) for p in train_paths + other_paths]
    token_counts = pd.DataFrame(index=cod, data=X, columns=tokens)
    token_counts.to_csv("legislative_proposal_token_counts.csv")

    # bullet_cod = dict(zip(list(labels.keys()), X[:, 98].tolist()))
    # tf-idf transformation
    X = tf_idf(X)
    cod = [get_cod(p) for p in train_paths + other_paths]
    token_tfidf = pd.DataFrame(index=cod, data=X, columns=tokens)
    token_tfidf.to_csv("legislative_proposal_token_tfidf.csv")

    X_train_sum = X[: len(train_sum)]
    X_other_sum = X[len(train_sum) :]
    # Split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_sum, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    # CV: exhaustive l2 parameter tuning
    penalty = ["l2"]
    C = np.logspace(-1, 1, 100)
    hparams = {"C": C, "penalty": penalty}
    model = LogisticRegression(
        random_state=RANDOM_SEED, solver="liblinear", max_iter=20
    )
    # stratified cross-val of hyperparams
    clf = GridSearchCV(model, hparams, cv=10, verbose=0, scoring="f1_macro")
    # model.fit(X_train, y)
    clf.fit(X_train, y_train)

    # best model
    print(clf.best_params_)
    print(f"Cross Validation f1_macro: {100*clf.best_score_:0.1f}%")
    x_preds = clf.predict(X_test)
    # print('Hold out subset accuracy: ', (x_preds == y_test).mean(), '%')
    print("Performance on held out subset from training data:")
    print(classification_report(y_test, x_preds))
    ConfusionMatrix(actual_vector=y_test, predict_vector=x_preds).print_matrix()

    clf.fit(X_train_sum, y)
    with open("lr.model.bin", "wb") as file:
        pickle.dump(clf, file)

    # # analyse tokens
    weights = clf.best_estimator_.coef_.reshape(-1)
    token_weights = pd.DataFrame(index=tokens, data=weights, columns=["Coefficient"])
    token_weights.to_csv("lr_word_coefficients.csv")

    weights_argsort = weights.argsort()
    tokens, token_weights = zip(*[(id2token[i], weights[i]) for i in weights_argsort])
    feat_importance = pd.DataFrame.from_dict(
        {"Tokens": tokens, "Weights": token_weights}
    )
    feat_importance.to_csv("lr_weights_by_tokens.csv")

    #
    x_other_preds = clf.predict(X_other_sum)
    x_other_probs = np.around(clf.predict_proba(X_other_sum)[:, 1], 2)

    cod = [get_cod(p) for p in other_paths]
    out = {
        "COD": cod,
        "LR_Predictions": x_other_preds.tolist(),
        "LR_Probabilities": x_other_probs.tolist(),
    }
    df = pd.DataFrame.from_dict(out)
    df = df.sort_values("COD", 0)
    df.to_csv("lr_output.csv")


if __name__ == "__main__":
    main()
