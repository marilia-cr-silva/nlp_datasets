# %%
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.metrics import balanced_accuracy_score

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
                'embedded_hs_13_multi_train.pkl']
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
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=5, cv=5,
                                    random_state=42, verbosity=2,
                                    scoring = "balanced_accuracy",
                                    max_time_mins = 15)
    pipeline_optimizer.fit(X_train, y_train)
    y_pred = pipeline_optimizer.predict(X_test)
    bal_acc_score = balanced_accuracy_score(y_true, y_pred)
    print(f"the balanced accuracy of {item_pair[0][9:-4]} is {bal_acc_score}")
    with open(f"tpot_bal_acc_{item_pair[0][9:-4]}.txt", "w") as new_file:
        new_file.write(f"{item_pair[0][9:-4]};{bal_acc_score};tpot\n")