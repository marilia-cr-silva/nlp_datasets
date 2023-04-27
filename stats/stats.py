import os
import re
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DirConfig:
    name: str
    relative_path: str

    def __str__(self):
        return f"name: `{self.name}` | relative_path: `{self.relative_path}`"


class StatsRunner:
    def __init__(self):
        self.VALID_TASKS = ["fn", "hs", "sa", "sd", "tc"]
        self.TASKS_RELATIVE_PATH = "../datasets/"
        self.WORKING_DIR = os.getcwd()

        self.tasks = self.get_tasks_dir_config()
        self.dataset_map = self.get_dataset_map()

        self.output_tasks = self.get_output_tasks_dir_config()
        self.output_dataset_map = self.get_output_dataset_map()

    def get_tasks_dir_config(self) -> List[DirConfig]:
        current_dir = os.getcwd()

        if current_dir != self.WORKING_DIR: 
            os.chdir(self.WORKING_DIR)

        tasks = [
            DirConfig(
                name=folder,
                relative_path=os.path.join(self.TASKS_RELATIVE_PATH, folder)
            )
            for folder in os.listdir(self.TASKS_RELATIVE_PATH)
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
                    relative_path=os.path.join(task.relative_path, ds)
                )
                for ds in os.listdir(task.relative_path) 
                if re.search("[a-z]{2}_[0-9]{2}", ds) != None
            ]
            dataset_map[task.name] = datasets

        return dataset_map
    
    def get_output_tasks_dir_config(self) -> List[DirConfig]:
        return [
            DirConfig(
                name=task.name,
                relative_path=os.path.join(self.WORKING_DIR, task.name)
            )
            for task in self.tasks
        ]
    
    def get_output_dataset_map(self) -> Dict[str, List[DirConfig]]:
        output_dataset_map = {}
        for task_name, datasets in self.dataset_map.items():
            output_dataset_map[task_name] = [
                DirConfig(
                    name=ds.name,
                    relative_path=os.path.join(self.WORKING_DIR, task_name, ds.name)
                )
                for ds in datasets
            ]
        return output_dataset_map

    def build_output_dir_structure(self) -> None:
        if os.getcwd() != self.WORKING_DIR: 
            os.chdir(self.WORKING_DIR)

        for task in self.output_tasks:
            if not os.path.isdir(task.relative_path):
                os.mkdir(task.relative_path)

        for task_name, datasets in self.output_dataset_map.items():
            for dataset in datasets:
                if not os.path.isdir(dataset.relative_path):
                    os.mkdir(dataset.relative_path)