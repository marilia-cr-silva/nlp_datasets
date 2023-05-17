# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Set

# %%
DATASET_STATS_COLUMNS = [
    "train_length",
    "test_length",
    "num_classes",
    "num_class_permutations",
    "label_imbalance_ratio",
    "dataset_name",
]
BOX_PLOT_COLUMNS = [
    "train_length",
    "test_length",
    "num_classes",
    "num_class_permutations",
    "label_imbalance_ratio",
]
TASK_STATS_COLUMNS = [
    "num_datasets",
    "total_class_permutations",
    "max_num_classes",
    "min_num_classes",
    "num_multiclass_datasets",
    "num_binary_datasets",
    "max_label_imbalance_ratio",
    "min_label_imbalance_ratio",
    "largest_train_set",
    "smallest_train_set",
    "largest_test_set",
    "smallest_test_set",
]
METRIC_TEXT_MAP = {
    "train_length": "Train Length",
    "test_length": "Test Length",
    "num_classes": "Number of Labels",
    "num_class_permutations": "Number of Label Permutations",
    "label_imbalance_ratio": "Label Imbalance Ratio",
}


# %%
def get_processed_files() -> List[str]:
    with open("proc_datasets.txt", "r") as f:
        all_files = [file.strip() for file in f.readlines()]
    return all_files


def build_dataset_file_map(files: List[str]) -> Dict[str, List[str]]:
    datasets = set([f[:5] for f in files])
    dataset_file_map = {}

    for dataset in datasets:
        current_files = [f for f in files if f.startswith(dataset)]
        dataset_file_map[dataset] = current_files

    return dataset_file_map


def build_task_dataset_map(file_dataset_map: Dict[str, List[str]]):
    tasks = ["fn", "hs", "sa", "sd"]

    task_dataset_map = {}

    for task in tasks:
        task_dataset_map[task] = {}

        current_datasets = [d for d in file_dataset_map.keys() if d.startswith(task)]

        for dataset in current_datasets:
            task_dataset_map[task][dataset] = file_dataset_map[dataset]

    return task_dataset_map


def run_stats(task_dataset_map: Dict[str, Any]):
    files_set = set(os.listdir())

    metrics_df_map = {metric: pd.DataFrame() for metric in BOX_PLOT_COLUMNS}
    tasks = task_dataset_map.keys()

    for task in tasks:
        dataset_stats_df = get_all_dataset_stats(task, task_dataset_map, files_set)

        if dataset_stats_df.empty:
            print(f"{task} yielded no dataset stat results. Skipping...")
            continue

        run_tasks_stats(task, dataset_stats_df)

        for metric in BOX_PLOT_COLUMNS:
            metrics_df_map[metric] = pd.concat(
                [metrics_df_map[metric], dataset_stats_df[metric]], axis=1
            )

    for metric in metrics_df_map.keys():
        metrics_df_map[metric].columns = tasks
        sns.boxplot(data=metrics_df[metric], orient="h", palette="Set2")

        plt.xlabel(METRIC_TEXT_MAP[metric])
        plt.figure(figsize=(8, 6))
        plt.savefig(f"boxplot_{metric}.png")

    return metrics_df_map


def get_all_dataset_stats(
    task_name: str, task_dataset_map: Dict[str, Any], files_set: Set[str]
):
    print(f"Running {task_name}...")

    dataset_results = pd.DataFrame(columns=DATASET_STATS_COLUMNS).astype(
        {
            "train_length": "int16",
            "test_length": "int16",
            "num_classes": "int16",
            "num_class_permutations": "int16",
            "label_imbalance_ratio": "float64",
            "dataset_name": "string",
        }
    )

    for dataset in task_dataset_map[task_name].keys():
        dataset_files = task_dataset_map[task_name][dataset]

        dataset_files_set = set(dataset_files)
        if len(dataset_files_set.intersection(files_set)) != len(dataset_files_set):
            print(f"{dataset} has missing files")
            continue

        result = run_dataset_stats(dataset, dataset_files)
        dataset_results = pd.concat(
            [dataset_results, result], axis=0, ignore_index=True
        )

    return dataset_results


def run_dataset_stats(dataset_name: str, dataset_files: List[str]):
    """
    All the datasets per task
    Every dataset:
    training set length and test set length
    Number of Classes
    Number of class permutations (n*(n-1)/2) -> n number of labels
    Imbalance ratio (
        number of the most represented label
        /
        number of the most underrepresented label)
        :
        (max(value_counts) / min(value_counts()
    )
    """
    print(f"Processing stats for {dataset_name}")
    stats_df = pd.DataFrame(
        {col: 0 for col in DATASET_STATS_COLUMNS}, index=[0]
    ).astype(
        {
            "train_length": "int16",
            "test_length": "int16",
            "num_classes": "int16",
            "num_class_permutations": "int16",
            "label_imbalance_ratio": "float64",
            "dataset_name": "string",
        }
    )
    multi_test_file_name = f"{dataset_name}_multi_test.csv"
    multi_train_file_name = f"{dataset_name}_multi_train.csv"

    if multi_test_file_name in dataset_files and multi_train_file_name in dataset_files:
        multi_test_df = pd.read_csv(multi_test_file_name, sep=";")
        multi_train_df = pd.read_csv(multi_train_file_name, sep=";")

        stats_df["test_length"] = multi_test_df.shape[0]
        stats_df["train_length"] = multi_train_df.shape[0]

        train_value_counts = multi_train_df["label"].value_counts()
        stats_df["label_imbalance_ratio"] = (
            train_value_counts.min() / train_value_counts.max()
        )
    else:
        test_df = pd.read_csv(f"{dataset_name}_bin_test_0_1.csv", sep=";")
        train_df = pd.read_csv(f"{dataset_name}_bin_train_0_1.csv", sep=";")

        stats_df["test_length"] = test_df.shape[0]
        stats_df["train_length"] = train_df.shape[0]

        train_value_counts = train_df["label"].value_counts()
        stats_df["label_imbalance_ratio"] = (
            train_value_counts.min() / train_value_counts.max()
        )

    explained_df = pd.read_csv(f"{dataset_name}_explained.csv", sep=";")
    num_classes = explained_df.shape[0]
    stats_df["num_classes"] = num_classes
    stats_df["num_class_permutations"] = int((num_classes * (num_classes - 1)) / 2)

    stats_df["dataset_name"] = dataset_name
    stats_df.to_csv(f"output/stats_{dataset_name}.csv", index=False, sep=";")
    print(f"Finished processing stats for {dataset_name}")

    return stats_df


def run_tasks_stats(task_name: str, datasets_df: pd.DataFrame):
    """
    Task level:
    Number of datasets: sum(num_files)
    Total class permutations: sum(num_class_permutations)
    Maximum number of classes: max(num_classes)
    Minimum number of classes: min(num_classes)
    Number of Multiclass datasets by default: use mask for num_classes > 2
    Number of Binary datasets by default: use mask for num_classes < 3
    Max imbalance ratio
    Min imbalance ratio
    Largest training set: sa_01
    Smallest training set: sa_12
    Largest test set: sa_05
    Smallest test set: sa_06
    """
    print(f"Processing task stats for {task_name}")
    task_df = pd.DataFrame(
        {col: 0 for col in TASK_STATS_COLUMNS},
        index=[0],
    ).astype(
        {
            "num_datasets": "int16",
            "total_class_permutations": "int16",
            "max_num_classes": "int16",
            "min_num_classes": "int16",
            "num_multiclass_datasets": "int16",
            "num_binary_datasets": "int16",
            "max_label_imbalance_ratio": "float64",
            "min_label_imbalance_ratio": "float64",
            "largest_train_set": "string",
            "smallest_train_set": "string",
            "largest_test_set": "string",
            "smallest_test_set": "string",
        }
    )
    task_df["num_datasets"] = datasets_df.shape[0]
    task_df["total_class_permutations"] = datasets_df["num_class_permutations"].sum()
    task_df["max_num_classes"] = datasets_df["num_classes"].max()
    task_df["min_num_classes"] = datasets_df["num_classes"].min()

    multi_class_datasets_mask = datasets_df["num_classes"] > 2
    task_df["num_multiclass_datasets"] = datasets_df.loc[
        multi_class_datasets_mask
    ].shape[0]
    task_df["num_binary_datasets"] = datasets_df.loc[~multi_class_datasets_mask].shape[
        0
    ]

    task_df["max_label_imbalance_ratio"] = datasets_df["label_imbalance_ratio"].max()
    task_df["min_label_imbalance_ratio"] = datasets_df["label_imbalance_ratio"].min()

    task_df["largest_train_set"] = datasets_df.iloc[
        datasets_df["train_length"].idxmax()
    ]["dataset_name"]
    task_df["smallest_train_set"] = datasets_df.iloc[
        datasets_df["train_length"].idxmin()
    ]["dataset_name"]

    task_df["largest_test_set"] = datasets_df.iloc[datasets_df["test_length"].idxmax()][
        "dataset_name"
    ]
    task_df["smallest_test_set"] = datasets_df.iloc[
        datasets_df["test_length"].idxmin()
    ]["dataset_name"]

    task_df.to_csv(f"output/stats_{task_name}.csv", sep=";", index=False)
    print(f"Finished processing task stats for {task_name}")


# %%
if __name__ == "__main__":
    files = get_processed_files()
    dataset_file_map = build_dataset_file_map(files)
    task_dataset_map = build_task_dataset_map(dataset_file_map)
    metrics_df = run_stats(task_dataset_map)
