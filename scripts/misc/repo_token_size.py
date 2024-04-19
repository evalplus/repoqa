# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

# Estimate the maximum token size for a repository

import json

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from repoqa.utility import topological_sort

COMMENT_PREFIX = {
    "python": "#",
    "java": "//",
    "typescript": "//",
    "rust": "//",
    "cpp": "//",
}


def get_max_token_size(dataset_path: str, model: str):
    with open(dataset_path, "r") as f:
        repo = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model)

    min_token_size = 1e9
    min_token_size_repo = None
    token_sizes = []

    for lang, repos in repo.items():
        for repo in repos:
            if "dependency" not in repo:
                print(
                    f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `dependency` -- do dependency analysis first"
                )
                continue

            ordered_paths = topological_sort(repo["dependency"])

            bigfile = ""
            for path in ordered_paths:
                bigfile += (
                    COMMENT_PREFIX[lang]
                    + f" current path: {path}\n"
                    + repo["content"][path]
                )

            # estimate the maximum token size
            token_size = tokenizer(bigfile, return_tensors="pt")["input_ids"].shape[1]
            token_sizes.append(token_size)
            min_token_size = min(min_token_size, token_size)
            if min_token_size == token_size:
                min_token_size_repo = repo["repo"]
            print(f"[{lang}] {repo['repo']:<32}: {token_size:>20} tokens")

    print(f"Estimated minimum token size: {min_token_size} by {min_token_size_repo}")

    # visualize the distribution
    plt.figure(figsize=(8, 4))
    plt.hist(token_sizes, bins=64)
    # xtick at every 20k
    unit = 100 * 1000
    plt.xticks(range(0, max(token_sizes) + 1, unit))
    plt.xlim(0, max(token_sizes) + 1000)
    # xtick using "k"
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / unit:.0f}")
    )
    plt.xlabel("Token size (100k)")
    plt.ylabel("Frequency")
    plt.title("Token size distribution")
    # compact layout
    plt.tight_layout()
    plt.savefig("token_size_distribution.png", dpi=164, bbox_inches="tight")


if __name__ == "__main__":
    from fire import Fire

    Fire(get_max_token_size)
