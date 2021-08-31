import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pycm import ConfusionMatrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tokenizers import Tokenizer

from src.learn_tokenizer import learn_tokenize
from src.utils import (filter_labels, filter_paths, get_cod, get_logger,
                       get_summary_paths, read_file, read_label,
                       tokenize_to_vec)
from src.weighting import tf_idf

log = get_logger(__name__)

"""
The sci-kit learn pipeline for scaling legislative summaries as crawled from the Legislative Observatory (https://oeil.secure.europarl.europa.eu/oeil/home/home.do).

The pipeline runs as follows:

Required preprocessing steps:
* Preprocess summaries to filter out irrelevant segments via regex
* Train a word-level tokenizer

Training and inference steps:
1. Loading and preparing data
2. Compute token counts
3. Compute term-frequency inverse-document frequency token weights
4. Run sklearn pipeline
    1. GridSearchCV to determine the best hyperparameters for L2-regularization, GridSearchCV by default
        * Holds out 20% of the training data for post-hoc evaluation
        * Optimizes L2 regularization in 10-fold cross-validation for which the regularization weight is sampled in logspace
    2. Evaluate on hold-out set to get estimate on performance
    3. Retrain model with full training data
5. Run scaling with full predictions on proposals and final acts

All steps throughout store relevant data in CSVs.

Abbreviations:
* `lp`: Legislative proposals
* `fa`: Final acts
* `train`: Annotated legislative proposals
* `other`: Legislative proposals without annotations
* `X`: tf-idf weighted token counts by `lp` or `fa`
* `config`: passed through experiment config via Hydra, see `config` folder
"""


def main(config: DictConfig):
    log.info(f"Starting experiment! Outputs will be saved at: {os.getcwd()}")
    # 1. LOADING DATA
    #   construct data file path from Hydra config
    data = Path(config.data_dir)
    #   read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
    labels, y = read_label(config.data.labels)
    #   get all pathlib.Paths to legislative proposal summaries

    lp_paths = get_summary_paths(
        os.path.join(config.summaries_dir, config.data.summaries), final_act=False
    )
    #   split Paths into (labelled, unlabelled) paths
    lp_train_paths, lp_other_paths = filter_labels(lp_paths, labels)

    #   Loading and tf-idf weighting data
    fa_paths = get_summary_paths(
        os.path.join(config.summaries_dir, config.data.summaries), final_act=True
    )
    #   split Paths into (labelled, unlabelled) paths
    fa_train_paths, fa_other_paths = filter_labels(fa_paths, labels)

    fa_sum = [read_file(x) for x in fa_paths]
    fa_cod = [get_cod(p) for p in fa_paths]

    #   load pre-trained huggingface tokenizer stored at ./data/tokenizers/classifier.json
    # tokenizer_path = data.joinpath("tokenizers").joinpath("classifier.json")
    # assert tokenizer_path.exists(), "Create tokenizer before training the model"
    tokenizer = learn_tokenize(
        lp_train_paths, data.joinpath("tokenizers").joinpath("classifier.json")
    )
    # tokenizer = Tokenizer.from_file(str(tokenizer_path))
    id2token = {
        v: k for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    }
    tokens = list(id2token.values())
    num_tokens = len(tokens)

    # 2. COMPUTE TOKEN COUNTS
    # read sentence-splitted legislative proposal summaries into List[List[str]]
    lp_train_sum = [read_file(x) for x in lp_train_paths]
    lp_other_sum = [read_file(x) for x in lp_other_paths]
    # list is ordered
    lp_all_sum = lp_train_sum + lp_other_sum
    # stack document-token count arrays, for instance
    #         word1, word2, ..., word n
    # doc1  [   1  , 10   , ...,   1    ]
    # doc2  [   0  ,  3   , ...,   1    ]
    # doc3  [   2  ,  5   , ...,   1    ]
    lp_token_counts = np.stack(
        [tokenize_to_vec(doc, tokenizer, num_tokens) for doc in lp_all_sum]
    )

    lp_train_other_cod = [get_cod(p) for p in lp_train_paths + lp_other_paths]
    lp_token_counts_df = pd.DataFrame(
        index=lp_train_other_cod, data=lp_token_counts, columns=tokens
    )
    lp_token_counts_df.to_csv("lp_token_counts.csv")

    # 3. COMPUTE TF IDF WEIGHTS
    #   Compute tf-idf weighted token counts for classifier
    X_lp_all, lp_idf = tf_idf(lp_token_counts, return_idf=True)
    lp_tfidf = pd.DataFrame(index=lp_train_other_cod, data=X_lp_all, columns=tokens)
    lp_tfidf.to_csv("lp_tfidf.csv")

    X_lp_train = X_lp_all[: len(lp_train_sum)]
    X_lp_other = X_lp_all[len(lp_train_sum) :]

    # 4. GRID SEARCH
    #   Split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_lp_train,
        y,
        test_size=config.hparams.test_split,
        random_state=config.seed,
        stratify=y,
    )
    # CV: exhaustive l2 parameter tuning
    C = np.logspace(
        config.hparams.regularization.log_space_start,
        config.hparams.regularization.log_space_end,
        config.hparams.regularization.log_space_samples,
    )
    penalty = [config.hparams.regularization.type]
    hparams = {"C": C, "penalty": penalty}
    model = hydra.utils.instantiate(config.model)
    # stratified cross-val of hyperparams
    clf = GridSearchCV(
        model,
        hparams,
        cv=config.hparams.cv.folds,
        verbose=config.hparams.cv.verbose,
        scoring=config.hparams.cv.scoring,
    )
    clf.fit(X_train, y_train)
    # clf.fit(X_lp_train, y)

    # best model
    log.info(
        f"The best found hyperparameters for your configuration: {clf.best_params_}"
    )
    log.info(
        f"The Cross-Validation {config.hparams.cv.scoring} for the hyperparameters: {100*clf.best_score_:0.1f}%"
    )
    X_test_preds = clf.predict(X_test)
    log.info("Performance on held out subset from training data:")
    log.info(classification_report(y_test, X_test_preds))
    ConfusionMatrix(actual_vector=y_test, predict_vector=X_test_preds).print_matrix()

    # model = hydra.utils.instantiate(config.model)
    # clf = GridSearchCV(
    #     model,
    #     hparams,
    #     cv=config.hparams.cv.folds,
    #     verbose=config.hparams.cv.verbose,
    #     scoring=config.hparams.cv.scoring,
    # )
    log.info(
        f"The best found hyperparameters for your configuration: {clf.best_params_}"
    )
    log.info(
        f"The Cross-Validation {config.hparams.cv.scoring} for the hyperparameters: {100*clf.best_score_:0.1f}%"
    )
    log.info("Retraining with full annotations!")
    clf.fit(X_lp_train, y)
    log.info(
        f"The best found hyperparameters for your configuration: {clf.best_params_}"
    )
    log.info(
        f"The Cross-Validation {config.hparams.cv.scoring} for the hyperparameters: {100*clf.best_score_:0.1f}%"
    )
    with open("classifier.model.bin", "wb") as file:
        pickle.dump(clf, file)
    log.info("Fully retrained and model stored!")

    # # analyse tokens
    if hasattr(clf.best_estimator_, "coef_"):
        weights = clf.best_estimator_.coef_.reshape(-1)
        token_weights = pd.DataFrame(
            index=tokens, data=weights, columns=["Coefficient"]
        )
        token_weights.to_csv("coef_by_token.csv")
        log.info("Coefficients by token stored!")

    x_other_preds = clf.predict(X_lp_other)
    x_other_probs = np.around(clf.predict_proba(X_lp_other)[:, 1], 2)

    lp_other_cod = [get_cod(p) for p in lp_other_paths]
    out = {
        "COD": lp_other_cod,
        "LR_Predictions": x_other_preds.tolist(),
        "LR_Probabilities": x_other_probs.tolist(),
    }
    scaling_df = pd.DataFrame.from_dict(out)
    scaling_df = scaling_df.sort_values("COD", 0)
    scaling_df.to_csv("lp_other_predictions.csv")

    # [5.] SCALING
    log.info("Starting inference! Loading files..")

    fa_token_counts = np.stack(
        [tokenize_to_vec(doc, tokenizer, num_tokens) for doc in fa_sum]
    )
    fa_token_counts_df = pd.DataFrame(
        index=fa_cod, data=fa_token_counts, columns=tokens
    )
    fa_token_counts_df.to_csv("fa_token_counts.csv")

    if config.hparams.recompute_idf:
        X_fa = tf_idf(fa_token_counts)
        log.info("Reweighted final act TF-IDF!")
    else:
        X_fa = tf_idf(fa_token_counts, idf=lp_idf)

    if config.hparams.test_lp:
        lp_validation_data = pd.read_csv(data.joinpath("lp_validation_data.csv"))
        validated_df = pd.merge(
            left=scaling_df,
            right=lp_validation_data.loc[:, ["COD", "LP_Coder_1", "LP_Coder_2"]],
            how="inner",
            on="COD",
        )
        ConfusionMatrix(
            actual_vector=validated_df["LP_Coder_1"].values,
            predict_vector=validated_df["LR_Predictions"].values,
        ).print_matrix()
        # import pudb
        # pu.db

    fa_tfidf = pd.DataFrame(index=fa_cod, data=X_fa, columns=tokens)
    fa_tfidf.to_csv(f"fa_tfidf_recompute-idf_{config.hparams.recompute_idf}.csv")

    #   Predicting estimates and classes for both proposals and final acts
    log.info("Files loaded.. predicting!")
    lp_preds = clf.predict(X_lp_all)
    lp_probs = np.around(clf.predict_proba(X_lp_all)[:, 1], 2)
    fa_preds = clf.predict(X_fa)
    fa_probs = np.around(clf.predict_proba(X_fa)[:, 1], 2)

    #   Get the overlapping paths
    overlapping_paths = list(set(lp_train_other_cod).intersection(set(fa_cod)))

    lp_paths = lp_train_paths + lp_other_paths
    #   Filter for overlapping paths and sort accordingly to align estimates and predictions
    filtered_lp_preds = filter_paths(lp_preds, lp_paths, overlapping_paths)
    filtered_lp_probs = np.array(filter_paths(lp_probs, lp_paths, overlapping_paths))
    filtered_fa_preds = filter_paths(fa_preds, fa_paths, overlapping_paths)
    filtered_fa_probs = np.array(filter_paths(fa_probs, fa_paths, overlapping_paths))
    #   Construct pandas DataFrame from data
    df_dico = {
        "COD": overlapping_paths,
        "LR_Proposal_Prediction": filtered_lp_preds,
        "LR_Proposal_Probability": filtered_lp_probs,
        "LR_Final_Act_Prediction": filtered_fa_preds,
        "LR_Final_Act_Probability": filtered_fa_probs,
        "LR_Same_Class": np.array(filtered_lp_preds) == np.array(filtered_fa_preds),
        "LR_Prob_dif": np.around(filtered_fa_probs - filtered_lp_probs, 2),
    }
    scaling_df = pd.DataFrame.from_dict(df_dico)
    scaling_df = scaling_df.sort_values("COD", axis=0)
    if config.hparams.test_fa:
        fa_validation_data = pd.read_csv(data.joinpath("fa_validation_data.csv"))
        validated_df = pd.merge(
            left=scaling_df,
            right=fa_validation_data.loc[:, ["COD", "FA_Coder_1", "FA_Coder_2"]],
            how="inner",
            on="COD",
        )
        # import pudb
        # pu.db
        ConfusionMatrix(
            actual_vector=validated_df["FA_Coder_1"].values,
            predict_vector=validated_df["LR_Final_Act_Prediction"].values,
        ).print_matrix()

    if (
        config.hparams.join_validation_data
        and config.hparams.test_lp
        and config.hparams.test_fa
    ):
        scaling_df = pd.merge(
            left=scaling_df,
            right=lp_validation_data.loc[:, ["COD", "LP_Coder_1", "LP_Coder_2"]],
            how="left",
            on="COD",
        )
        scaling_df = pd.merge(
            left=scaling_df,
            right=fa_validation_data.loc[:, ["COD", "FA_Coder_1", "FA_Coder_2"]],
            how="left",
            on="COD",
        )
        scaling_df.to_csv("lp_fa_scaling.csv", index=False)
    log.info("Predictions stored")
