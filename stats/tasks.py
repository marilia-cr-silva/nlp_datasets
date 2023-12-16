# %%
import os
import re
import shutil
import traceback
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DirConfig:
    name: str
    absolute_path: str

    def __str__(self):
        return f"name: `{self.name}` | absolute_path: `{self.absolute_path}`"


# %%
class TasksRunner:
    def __init__(self):
        self.VALID_TASKS = ["fn", "hs", "sa", "sd"]
        self.WORKING_DIR = os.getcwd()
        self.TASKS_RELATIVE_PATH = "../datasets/"
        self.TASKS_ABSOLUTE_PATH = self.get_tasks_absolute_path()

        self.tasks = self.get_tasks_dir_config()
        self.dataset_map = self.get_dataset_map()

        self.output_tasks = self.get_output_tasks_dir_config()
        self.output_dataset_map = self.get_output_dataset_map()

        self.processed_datasets = self.get_processed_datasets()

    def get_tasks_absolute_path(self):
        if os.getcwd() != self.TASKS_RELATIVE_PATH:
            os.chdir(self.TASKS_RELATIVE_PATH)
        return os.path.abspath(os.getcwd())

    def get_tasks_dir_config(self) -> List[DirConfig]:
        current_dir = os.getcwd()

        if current_dir != self.WORKING_DIR:
            os.chdir(self.WORKING_DIR)

        tasks = [
            DirConfig(
                name=folder,
                absolute_path=os.path.join(self.TASKS_ABSOLUTE_PATH, folder)
            )
            for folder in os.listdir(self.TASKS_ABSOLUTE_PATH)
            if folder in self.VALID_TASKS
        ]

        print(f"Found the following tasks: {tasks}")

        return tasks

    def get_dataset_map(self) -> Dict[str, List[DirConfig]]:
        dataset_map = {}

        for task in self.tasks:
            datasets = [
                DirConfig(
                    name=ds,
                    absolute_path=os.path.join(task.absolute_path, ds)
                )
                for ds in os.listdir(task.absolute_path)
                if re.search("[a-z]{2}_[0-9]{2}", ds) != None
            ]
            dataset_map[task.name] = datasets

        return dataset_map

    def get_output_tasks_dir_config(self) -> List[DirConfig]:
        return [
            DirConfig(
                name=task.name,
                absolute_path=os.path.join(self.WORKING_DIR, task.name)
            )
            for task in self.tasks
        ]

    def get_output_dataset_map(self) -> Dict[str, List[DirConfig]]:
        output_dataset_map = {}
        for task_name, datasets in self.dataset_map.items():
            output_dataset_map[task_name] = [
                DirConfig(
                    name=ds.name,
                    absolute_path=os.path.join(
                        self.WORKING_DIR, task_name, ds.name)
                )
                for ds in datasets
            ]
        return output_dataset_map

    def build_output_dir_structure(self) -> None:
        if os.getcwd() != self.WORKING_DIR:
            os.chdir(self.WORKING_DIR)

        for task in self.output_tasks:
            if not os.path.isdir(task.absolute_path):
                os.mkdir(task.absolute_path)

        for task_name, datasets in self.output_dataset_map.items():
            for dataset in datasets:
                if not os.path.isdir(dataset.absolute_path):
                    os.mkdir(dataset.absolute_path)

    def find_corresponding_dataset(self, output_dataset_name: str, datasets: List[DirConfig]) -> DirConfig:
        for dataset in datasets:
            if dataset.name == output_dataset_name:
                return dataset
        raise ValueError(
            f"Could not find corresponding dataset for {dataset.name}")

    def get_processed_datasets(self):
        with open("processed_files.txt", "r") as f:
            processed_datasets = f.readlines()
        processed_datasets = set([ds[:5] for ds in processed_datasets])
        print(f"Found {len(processed_datasets)} already processed datasets.")
        return processed_datasets

    def run_tasks(self):
        os.chdir(self.WORKING_DIR)
        self.build_output_dir_structure()

        for task in self.tasks:
            output_task = [
                ot for ot in self.output_tasks if ot.name == task.name][0]

            datasets = self.dataset_map[task.name]
            output_datasets = self.output_dataset_map[task.name]

            failed = {}

            for output_dataset in output_datasets:
                if output_dataset.name in self.processed_datasets:
                    print(
                        f"{output_dataset.name} is already processed. Skipping...")
                    continue

                dataset = self.find_corresponding_dataset(
                    output_dataset.name, datasets)

                os.chdir(dataset.absolute_path)
                python_script_path = os.path.join(
                    dataset.absolute_path, f"{dataset.name}.py")

                try:
                    with open(python_script_path, 'r') as f:
                        print(f"executing {python_script_path}")
                        exec(f.read(), locals(), locals())
                except:
                    format_exc = traceback.format_exc()

                    print(f"{python_script_path} failed:\n{format_exc}")

                    failed[task.name] = format_exc

                # Get all output files generated in current dir, ignoring current script
                output_files = [f for f in os.listdir(
                ) if re.search("\.csv$", f) != None]

                for output_file in output_files:
                    print(
                        f"moving {output_file} to {output_dataset.absolute_path}")
                    shutil.move(output_file, output_dataset.absolute_path)
