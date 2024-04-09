# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from random import sample

from tqdm import tqdm


# Annotate an incomplete repoqa dataset with function and class information
def main(
    dataset_path: str,
    overwrite_analysis: bool = False,
    max_len: int = 2000,
    max_fn_per_repo: int = 10,
):
    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    with open(dataset_path, "r") as f:
        lists = json.load(f)

    for lang, repos in lists.items():
        print(f"🔥 Selecting needle functions for {lang}")
        # FIXME(@ganler): enable more dependency analysis!
        for repo in tqdm(repos):
            # skip if the repo already has function information
            if not overwrite_analysis and repo.get("needles"):
                continue

            if not repo.get("functions"):
                print(
                    f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `functions` field -- do function analysis first"
                )
                continue

            selected_bins = set()

            function_names = Counter()
            for funcs in repo["functions"].values():
                for fn in funcs:
                    function_names.update([fn["name"]])
            # get function names that only appear once
            function_names = {k for k, v in function_names.items() if v == 1}

            needle_candidates = []
            for path, funcs in repo["functions"].items():
                for fn in funcs:
                    # criteria 1: no repeated function names
                    if fn["name"] not in function_names:
                        continue

                    # criteria 2: length <= max_len
                    if fn["end_byte"] - fn["start_byte"] > max_len:
                        continue

                    # criteria 3: not in the same bin
                    bin_idx = fn["global_start_byte"] // max_len
                    if bin_idx in selected_bins:
                        continue

                    # criteria 4: TODO -- select those with more code!
                    selected_bins |= {bin_idx}
                    needle_candidates.append((path, fn))

            print(
                f"Selected {len(needle_candidates)} needles from {len(repo['functions'])} functions in {repo['repo']} ({lang})"
            )
            needles = []
            for path, fn in sample(
                needle_candidates, min(max_fn_per_repo, len(needle_candidates))
            ):
                needles.append({**fn, "path": path})
            repo["needles"] = needles

    # update the dataset
    with open(dataset_path, "w") as f_out:
        json.dump(lists, f_out)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
