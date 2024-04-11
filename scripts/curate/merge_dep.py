# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

from fire import Fire


def main(dataset_path: str):
    # iterate json files under scripts/curate/dep_analysis/data
    repo2dep = {}
    for file in os.listdir(os.path.join("scripts/curate/dep_analysis/data")):
        if file.endswith(".json"):
            with open(os.path.join("scripts/curate/dep_analysis/data", file)) as f:
                data = json.load(f)

            repos = list(data.values())[0]
            for repo in repos:
                repo2dep[repo["repo"]] = repo["dependency"]

    with open(dataset_path) as f:
        dataset = json.load(f)

    for lang, repos in dataset.items():
        for repo in repos:
            if repo["repo"] not in repo2dep:
                print(f"{lang} -- Repo {repo['repo']} not found in dep analysis data")
                continue
            repo["dependency"] = repo2dep[repo["repo"]]
            print(f"{lang} -- Repo {repo['repo']} has dependency added in the dataset")

    with open(dataset_path, "w") as f_out:
        json.dump(dataset, f_out)


if __name__ == "__main__":
    Fire(main)
