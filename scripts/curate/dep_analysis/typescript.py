# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import git
import tempdir
from fire import Fire
from tqdm.auto import tqdm

from scripts.curate.dataset_ensemble_clone import get_files_to_include
from scripts.curate.utility import lang2suffix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def traverse_dep(
    current_tree: Dict, current_file: str, entrypoint: str, dependencies: Dict
) -> None:
    if current_tree == None:
        dependencies[current_file] = set()
        return

    results = set(
        [
            file
            for file in current_tree.keys()
            if entrypoint in file and Path(file).suffix in [".ts", ".js"]
        ]
    )
    dependencies[current_file] = results

    for result in results:
        traverse_dep(current_tree[result], result, entrypoint, dependencies)


def add_circular(circular_dependencies: Dict, dependencies: Dict):
    for cycle in circular_dependencies:
        for index, from_file in enumerate(cycle[:-1]):
            to_file = cycle[index + 1]
            dependencies[from_file].add(to_file)


def get_dependencies(
    file_path: Path, temp_dir: str, entrypoint: str, dependencies: Dict
) -> None:
    command_list = f"dep-tree tree {file_path} --json".split()
    output_string = subprocess.check_output(command_list, cwd=temp_dir)
    output_string = output_string.decode("utf-8")[1:-2].split()
    json_string = "".join(chr(int(value)) for value in output_string)

    json_data = json.loads(json_string)
    tree_data = json_data["tree"]
    circular_dependencies = json_data["circularDependencies"]
    current_file = list(tree_data.keys())[0]

    traverse_dep(tree_data[current_file], current_file, entrypoint, dependencies)
    add_circular(circular_dependencies, dependencies)


# dataset_path is the dataset generated by dataset_ensemble_clone.py
def main():
    with open("scripts/cherrypick/lists.json") as f:
        lists = json.load(f)

    lang_suffix = lang2suffix["typescript"]
    repos = lists["typescript"]
    for repo in tqdm(repos):
        repo_name = repo["repo"]
        commit_sha = repo["commit_sha"]
        entrypoint = repo["entrypoint_path"]
        print(f"Visiting https://github.com/{repo_name}/tree/{commit_sha}")

        dependencies = {}
        with tempdir.TempDir() as temp_dir:
            gh_repo = git.Repo.clone_from(
                f"https://github.com/{repo_name}.git",
                temp_dir,
            )
            gh_repo.git.checkout(commit_sha)
            abs_prefix = Path(os.path.join(temp_dir, entrypoint))
            if "/" in entrypoint:
                suffix_path = entrypoint.split("/")[-1]
            else:
                suffix_path = entrypoint
            for file_path in abs_prefix.rglob("*"):
                if file_path.is_file() and file_path.suffix in lang_suffix:
                    current_name = os.path.relpath(str(file_path), temp_dir)
                    if current_name in dependencies:
                        continue
                    get_dependencies(file_path, temp_dir, suffix_path, dependencies)

            for key, value in dependencies.items():
                dependencies[key] = list(value)

            if "/" in entrypoint:
                updated_dependencies = {}
                append_prefix = "/".join(entrypoint.split("/")[:-1]) + "/"
                for key, values in dependencies.items():
                    new_key = append_prefix + key
                    if not ((Path(temp_dir) / Path(new_key)).exists()):
                        continue
                    new_values = []
                    for value in values:
                        new_value = append_prefix + value
                        if (Path(temp_dir) / Path(new_value)).exists():
                            new_values.append(new_value)
                    updated_dependencies[new_key] = new_values
                dependencies = updated_dependencies
            repo["dependency"] = dependencies
    with open(os.path.join(CURRENT_DIR, "data", "typescript.json"), "w") as f_out:
        json.dump({"typescript": repos}, f_out)


if __name__ == "__main__":
    Fire(main)
