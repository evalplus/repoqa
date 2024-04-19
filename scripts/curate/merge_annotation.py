# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json

from fire import Fire


def main(dataset_path: str, annotation_path: str):
    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    assert annotation_path.endswith(
        ".jsonl"
    ), "Annotation must be a JSONL file, check README"

    with open(dataset_path) as f:
        dataset = json.load(f)

    with open(annotation_path) as f:
        annotations = [json.loads(line) for line in f]

    def make_key(repo_name, func_name):
        return f"{repo_name}::{func_name}"

    key2annotation = {make_key(a["repo"], a["name"]): a for a in annotations}

    for lang, repos in dataset.items():
        for repo in repos:
            if "needles" not in repo:
                print(
                    f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `needles` -- do needle analysis first"
                )
                continue
            for needle in repo["needles"]:
                needle_name = needle["name"]
                key = make_key(repo["repo"], needle_name)
                annotation = key2annotation.get(key, None)
                if annotation is None:
                    print(
                        f"⚠️ Missing annotation for {key} for lang {lang} -- skipping"
                    )
                    continue
                needle["description"] = annotation["annotation"]

    with open(dataset_path, "w") as f_out:
        json.dump(dataset, f_out)


if __name__ == "__main__":
    Fire(main)
