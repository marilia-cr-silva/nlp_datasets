# %%
"""
This .py file runs the AutoGluon AutoML System

@article{agtabular,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}
"""
# %%
import os
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import balanced_accuracy_score

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

for item_pair in list_files:
    print(f"item 0: {item_pair[0]} and item 1: {item_pair[1]}")
    df_test = pd.read_pickle(item_pair[0])
    df_train = pd.read_pickle(item_pair[1])
    X_train = pd.DataFrame(np.array(df_train["text"].tolist()))
    X_train = pd.concat([X_train, df_train["label"]], axis=1)
    X_test = pd.DataFrame(np.array(df_test["text"].tolist()))
    y_true = np.array(df_test["label"])

    unique_labels = df_train["label"].unique()
    number_unique_label = len(unique_labels)

    if number_unique_label == 2:
        type_of_problem = "binary"
    else:
        type_of_problem = "multiclass"

    predictor = TabularPredictor(label="label", problem_type=type_of_problem)
    predictor.fit(X_train, time_limit=900, presets='high_quality')
    y_pred = predictor.predict(X_test)
    bal_acc_score = balanced_accuracy_score(y_true, y_pred)
    print(f"the balanced accuracy of {item_pair[0][9:-4]} is {bal_acc_score}")
    with open(f"autogluon_bal_acc_{item_pair[0][9:-4]}.txt", "w") as new_file:
        new_file.write(f"{item_pair[0][9:-4]};{bal_acc_score};autogluon\n")
