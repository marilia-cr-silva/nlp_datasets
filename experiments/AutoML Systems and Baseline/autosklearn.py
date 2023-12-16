# %%
"""
@inproceedings{autosklearn_2015,
author = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina and Springenberg, Jost Tobias and Blum, Manuel and Hutter, Frank},
title = {Efficient and Robust Automated Machine Learning},
year = {2015},
booktitle = {Proceedings of the 28th International Conference on Neural Information Processing Systems},
pages = {2755–2763},
}
"""
# %%
import os
import autosklearn.classification
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
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

# %%
for item_pair in list_files:
    print(f"item 0: {item_pair[0]} and item 1: {item_pair[1]}")
    df_test = pd.read_pickle(item_pair[0])
    df_train = pd.read_pickle(item_pair[1])
    X_train = np.array(df_train["text"].tolist())
    y_train = np.array(df_train["label"].tolist())
    X_test = np.array(df_test["text"].tolist())
    y_true = np.array(df_test["label"].tolist())

    automl_sklearn = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=900)

    automl_sklearn.fit(X_train, y_train)
    y_pred = automl_sklearn.predict(X_test)
    bal_acc_score = balanced_accuracy_score(y_true, y_pred)
    try:
      f1_calc_score = f1_score(y_true, y_pred, average="macro")
    except Exception:
      f1_calc_score = 0
    acc_score = accuracy_score(y_true, y_pred)
    print(f"the balanced accuracy of {item_pair[0][9:-4]} is {bal_acc_score}")
    with open(f"autosklearn_bal_acc_{item_pair[0][9:-4]}.txt", "w") as new_file:
        new_file.write(
            f"{item_pair[0][9:-4]}; acc: {acc_score}; bal acc: {bal_acc_score}; f1-score: {f1_calc_score}; autosklearn\n"
        )
