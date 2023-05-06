# %%
import matplotlib
import os
import pandas as pd
import re

# %%
regex = r"stats_([a-zA-z]{2})_\d{2}\.csv"

files = [f for f in os.listdir() if re.search(regex, f) != None]

DATASET_STATS_COLUMNS = [
    "train_length",
    "test_length",
    "num_classes",
    "num_class_permutations",
    "label_imbalance_ratio",
    "dataset_name",
]

tasks = ["hs", "sa", "sd", "fn"]
task_df_map = {}
for task in tasks:
    current_task_files = [f for f in files if task in f]
    task_df = pd.DataFrame(columns=DATASET_STATS_COLUMNS)
    for f in current_task_files:
        task_df = pd.concat(
            [task_df, pd.read_csv(f, sep=';')],
            axis=0,
            ignore_index=True
        )
    task_df_map[task] = task_df

# %%
boxplot_metric_map = {}
for col in DATASET_STATS_COLUMNS:
    if col == "dataset_name":
        continue
    boxplot_metric_map[col] = pd.DataFrame({ds: [] for ds in tasks})

    for task in tasks:
        boxplot_metric_map[col][task] = (
            task_df_map[task][col].astype('float64')
        )

