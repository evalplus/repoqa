# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime

import git
import tempdir
from fire import Fire
from tqdm.auto import tqdm

from scripts.curate.utility import lang2suffix


def get_files_to_include(gh_repo, entrypoint, lang_suffix):
    files_to_include = []
    for entry in gh_repo.commit().tree.traverse():
        if entry.path.startswith(entrypoint) and any(
            [entry.path.endswith(suffix) for suffix in lang_suffix]
        ):
            files_to_include.append((entry.path, entry.abspath))
    return files_to_include


def main(
    target_path: str = f"repoqa-{datetime.now().isoformat()}.json",
):
    # read /scripts/cherrypick/lists.json
    with open("scripts/cherrypick/lists.json") as f:
        lists = json.load(f)

    for lang, repos in lists.items():
        lang_suffix = lang2suffix[lang]
        for repo in tqdm(repos):
            repo_name = repo["repo"]
            commit_sha = repo["commit_sha"]
            entrypoint = repo["entrypoint_path"]

            print(f"Visiting https://github.com/{repo_name}/tree/{commit_sha}")

            if repo.get("content"):
                print(f"Skipping {repo_name} as it already has content.")
                continue

            with tempdir.TempDir() as temp_dir:
                gh_repo = git.Repo.clone_from(
                    f"https://github.com/{repo_name}.git",
                    temp_dir,
                )
                gh_repo.git.checkout(commit_sha)

                files_to_include = get_files_to_include(
                    gh_repo, entrypoint, lang_suffix
                )

                repo["content"] = {}
                for path, abspath in files_to_include:
                    with open(abspath, "r") as f:
                        repo["content"][path] = f.read()
    with open(target_path, "w") as f_out:
        json.dump(lists, f_out)


if __name__ == "__main__":
    Fire(main)
