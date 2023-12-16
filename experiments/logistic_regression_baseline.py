"""
This .py file has the logistic regression baseline
"""

# %% loading libraries
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# %% creating search spaces
def get_search_space_hpo_logistic_regression() -> dict:
    """
    it returns the search space for the logistic regression

    output:
    dict_search_space (dict):
    it has the search space for the classifier
    """
    dict_search_space = {
        "penalty": ["l2", "none"],
        "C": [0, 0.5, 1, 1.5, 2],
        "random_state": [0, 5, 10, 20, 30, 42, 100],
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "multi_class": ["auto", "ovr"]
    }
    return dict_search_space


# %% loading training and test sets
list_files = [['embedded_hs_01_bin_test_0_1.pkl',
               'embedded_hs_01_bin_train_0_1.pkl'],
              ['embedded_hs_02_multi_test.pkl',
               'embedded_hs_02_multi_train.pkl'],
              ["embedded_hs_03_bin_test_0_1.pkl",
               "embedded_hs_03_bin_train_0_1.pkl"],
              ['embedded_hs_04_bin_test_0_1.pkl',
               'embedded_hs_04_bin_train_0_1.pkl'],
              ['embedded_hs_05_multi_test.pkl',
               'embedded_hs_05_multi_train.pkl'],
              ['embedded_hs_06_bin_test_0_1.pkl',
               'embedded_hs_06_bin_train_0_1.pkl'],
              ['embedded_hs_07_bin_test_0_1.pkl',
               'embedded_hs_07_bin_train_0_1.pkl'],
              ['embedded_hs_08_bin_test_0_1.pkl',
               'embedded_hs_08_bin_train_0_1.pkl'],
              ['embedded_hs_09_bin_test_0_1.pkl',
               'embedded_hs_09_bin_train_0_1.pkl'],
              ['embedded_hs_10_bin_test_0_1.pkl',
               'embedded_hs_10_bin_train_0_1.pkl'],
              ['embedded_hs_11_bin_test_0_1.pkl',
               'embedded_hs_11_bin_train_0_1.pkl'],
              ['embedded_hs_12_bin_test_0_1.pkl',
               'embedded_hs_12_bin_train_0_1.pkl'],
              ['embedded_hs_13_multi_test.pkl',
               'embedded_hs_13_multi_train.pkl'],
              ["embedded_hs_14_bin_test_0_1.pkl",
               "embedded_hs_14_bin_train_0_1.pkl"],
              ["embedded_hs_15_multi_test.pkl",
               "embedded_hs_15_multi_train.pkl"],              
              ]

for item_pair in list_files:
    print(f"item 0: {item_pair[0]} and item 1: {item_pair[1]}")
    df_test = pd.read_pickle(item_pair[0])
    df_train = pd.read_pickle(item_pair[1])
    X_train = np.array(df_train["text"].tolist())
    y_train = np.array(df_train["label"])
    X_test = np.array(df_test["text"].tolist())
    y_true = np.array(df_test["label"])
    dict_lr = get_search_space_hpo_logistic_regression()
    logistic = LogisticRegression(
        warm_start=False,
        n_jobs=-1,
        l1_ratio=None,
        verbose=0,
        max_iter=100,
        dual=False,
        tol=10**-4,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
    )
    clf = RandomizedSearchCV(
        estimator=logistic,
        param_distributions=dict_lr,
        random_state=42,
        scoring="balanced_accuracy",
    )  # cv=5 by default
    r_search = clf.fit(X_train, y_train)
    y_pred = r_search.predict(X_test)

    bal_acc_score = balanced_accuracy_score(y_true, y_pred)
    try:
      f1_calc_score = f1_score(y_true, y_pred, average="macro")
    except Exception:
      f1_calc_score = 0
    acc_score = accuracy_score(y_true, y_pred)
    print(f"the balanced accuracy of {item_pair[0][9:-4]} is {bal_acc_score}")
    with open(f"logistic_regression_bal_acc_{item_pair[0][9:-4]}.txt", "w") as new_file:
        new_file.write(
            f"{item_pair[0][9:-4]}; acc: {acc_score}; bal acc: {bal_acc_score}; f1-score: {f1_calc_score}; logistic_regression\n"
        )
