# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from typing import Dict, List

import git
import tempdir
from fire import Fire
from tqdm.auto import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def remove_relative(path: str) -> str:
    parts = path.split(os.sep)
    filtered_parts = [part for part in parts if part not in (".", "..")]
    new_path = os.path.join(*filtered_parts)
    return new_path


def sanitize_paths(data: Dict[str, List[str]], entrypoint: str) -> Dict[str, List[str]]:
    sanitized_data = {}
    for file, dependencies in data.items():
        updated_file = os.path.join(entrypoint, remove_relative(file))
        updated_dependencies = []
        for dependency in dependencies:
            updated_dependency = os.path.join(entrypoint, remove_relative(dependency))
            updated_dependencies.append(updated_dependency)
        sanitized_data[updated_file] = updated_dependencies
    return sanitized_data


def run_dependency_analysis(config_file, go_file):
    # Load the JSON configuration
    with open(config_file, "r") as file:
        data = json.load(file)

    repos = data["go"]

    # Iterate over each repo entry in the JSON configuration
    for entry in tqdm(repos):
        repo_name = entry["repo"]
        commit_sha = entry["commit_sha"]
        entrypoint_path = entry["entrypoint_path"]

        print(f"Visiting https://github.com/{repo_name}/tree/{commit_sha}")

        with tempdir.TempDir() as temp_dir:
            gh_repo = git.Repo.clone_from(
                f"https://github.com/{repo_name}.git",
                temp_dir,
            )
            gh_repo.git.checkout(commit_sha)
            shutil.copy(go_file, temp_dir)

            command_list = (
                f"go build -o dependency_analysis dependency_analysis.go".split()
            )
            subprocess.run(command_list, cwd=temp_dir)

            command_list = f"./dependency_analysis {entrypoint_path} {repo_name.split('/')[-1]}".split()

            subprocess.run(command_list, cwd=temp_dir)
            output_dir = os.path.join(temp_dir, "output.json")
            with open(output_dir, "r") as output_file:
                output_data = json.load(output_file)
                entry["dependency"] = sanitize_paths(
                    output_data["repoName"], entrypoint_path
                )

    # Write all output data to a file
    with open(os.path.join(CURRENT_DIR, "data", "go.json"), "w") as f_out:
        json.dump({"go": repos}, f_out)


def main():
    config_file = "scripts/cherrypick/lists.json"
    go_file = "scripts/curate/dep_analysis/go-analysis/dependency_analysis.go"

    run_dependency_analysis(config_file, go_file)


if __name__ == "__main__":
    Fire(main)
