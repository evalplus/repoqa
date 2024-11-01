# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

from fire import Fire


# TODO: Have a unified way to do this once/if this scales up
def clean_path(path: str) -> str:
    return os.path.basename(path)


def main(dataset_path: str):
    # iterate json files under scripts/curate/dep_analysis/data
    repo2callgraph = {}
    for file in os.listdir(os.path.join("scripts/curate/interprocedural/data")):
        if file.endswith(".json"):
            with open(os.path.join("scripts/curate/interprocedural/data", file)) as f:
                data = json.load(f)

            repos = list(data.values())[0]
            for repo in repos:
                # TODO: cleanup
                if not "call_graph" in repo:
                    continue
                repo2callgraph[repo["repo"]] = repo["call_graph"]

    with open(dataset_path) as f:
        dataset = json.load(f)

    for lang, repos in dataset.items():
        # TODO: remove
        if lang != "python":
            continue

        for repo in repos:
            merged_functions = 0
            export_graph = {}
            if not repo["repo"] in repo2callgraph:
                continue
            current_graph = repo2callgraph[repo["repo"]]

            # TODO: Unify this (for python use full path in call graph)
            mapping = {}
            for _, functions in repo["functions"].items():
                for function in functions:
                    base_name = clean_path(function["file"])
                    class_type = function["class_type"]
                    function_name = function["name"]
                    if function["class_type"] == "":
                        check_name = base_name + "::" + function_name
                    else:
                        check_name = base_name + "::" + class_type + "." + function_name
                    mapping[check_name] = function["full_name"]
            visited = set()
            for file, functions in repo["functions"].items():
                for function in functions:
                    base_name = clean_path(function["file"])
                    class_type = function["class_type"]
                    function_name = function["name"]
                    if function["class_type"] == "":
                        check_name = base_name + "::" + function_name
                    else:
                        check_name = base_name + "::" + class_type + "." + function_name
                    if check_name in current_graph:
                        visited.add(check_name)
                        function["targets"] = list(current_graph[check_name]["targets"])
                        # TODO: Use full path
                        for index, val in enumerate(function["targets"]):
                            function["targets"][index] = mapping[val]
                        export_graph[function["full_name"]] = function
                        merged_functions += 1
            repo["call_graph"] = export_graph
            print(f"🎉 Merged {merged_functions} functions in {repo['repo']} ({lang})")

    with open(dataset_path, "w") as f_out:
        json.dump(dataset, f_out)


if __name__ == "__main__":
    Fire(main)
