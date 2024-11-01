# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess

import git
import tempdir
from fire import Fire
from tqdm.auto import tqdm

from scripts.curate.dataset_ensemble_clone import get_files_to_include
from scripts.curate.utility import lang2suffix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CALL_CYCLE_FILE = "call-cycle.json"

# Byproduct of code2flow tool, need to add file extension to full name
def convert_python_path(full_name) -> str:
    name_parts = full_name.split("::")
    file_name = name_parts[0] + ".py"

    if "." in name_parts[1]:
        class_name = name_parts[1].split(".")[0]
        function_name = name_parts[1].split(".")[1]
        return file_name + "::" + class_name + "." + function_name
    class_name = ""
    function_name = name_parts[1]
    return file_name + "::" + function_name


def main():
    with open("scripts/cherrypick/lists.json") as f:
        lists = json.load(f)

    repos = lists["python"]
    for repo in tqdm(repos):
        repo_name = repo["repo"]
        commit_sha = repo["commit_sha"]
        entrypoint = repo["entrypoint_path"]
        print(f"Visiting https://github.com/{repo_name}/tree/{commit_sha}")

        # TODO: one repo isn't parsable atm bug: https://github.com/scottrogowski/code2flow/issues/81, need to deal with this
        if "marshmallow" in repo_name:
            continue

        with tempdir.TempDir() as temp_dir:
            gh_repo = git.Repo.clone_from(
                f"https://github.com/{repo_name}.git",
                temp_dir,
            )
            gh_repo.git.checkout(commit_sha)

            # Create temporary file to store output
            with open(os.path.join(temp_dir, CALL_CYCLE_FILE), "a+") as file:
                command_list = f"code2flow {entrypoint} --output {file.name} --language py --no-trimming".split()

                _ = subprocess.run(command_list, cwd=temp_dir, capture_output=True)
                result = json.load(file)

            call_graph = result["graph"]
            nodes = call_graph["nodes"]
            edges = call_graph["edges"]

            graph_result = {}
            mapping = {}

            for id, node in nodes.items():
                full_name = convert_python_path(node["name"])
                mapping[id] = full_name

                name_parts = full_name.split("::")
                file_name = name_parts[0]

                if "." in name_parts[1]:
                    class_name = name_parts[1].split(".")[0]
                    function_name = name_parts[1].split(".")[1]
                else:
                    class_name = ""
                    function_name = name_parts[1]

                graph_result[full_name] = {
                    "file_name": file_name,
                    "class_name": class_name,
                    "function_name": function_name,
                    "targets": [],
                }

            for edge in edges:
                source = mapping[edge["source"]]
                target = mapping[edge["target"]]
                graph_result[source]["targets"].append(target)

            repo["call_graph"] = graph_result

    with open(
        os.path.join(CURRENT_DIR, "data", "python-call-graph.json"), "w"
    ) as f_out:
        json.dump({"python": repos}, f_out)


if __name__ == "__main__":
    Fire(main)
