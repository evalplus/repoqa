# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

"""
! Note: not fully usable. You might encounter:
github.GithubException.GithubException: 422 {"message": "Validation Failed", "errors": [{"message": "The listed users and repositories cannot be searched either because the resources do not exist o
r you do not have permission to view them.", "resource": "Search", "field": "q", "code": "invalid"}], "documentation_url": "https://docs.github.com/v3/search/"}
"""

import json
import os
from datetime import datetime
from typing import TypedDict

from fire import Fire
from github import Auth, Github
from github.Repository import Repository
from tqdm.auto import tqdm

from scripts.curate.utility import lang2suffix


class GitHubRepoMeta(TypedDict):
    repo_name: str
    repo_owner: str
    commit_sha: str
    repo_size: int


class GitHubDocument(GitHubRepoMeta):
    timestamp: str
    path: str
    content: str


def main(
    target_path: str = f"repoqa-{datetime.now().isoformat()}.json",
):
    token = os.getenv("GITHUB_TOKEN")
    assert token is not None, "Make a token at https://github.com/settings/tokens"
    auth = Auth.Token(token)

    # read /scripts/cherrypick/lists.json
    with open("scripts/cherrypick/lists.json") as f:
        lists = json.load(f)

    g = Github(auth=auth, per_page=1)
    for lang, repos in lists.items():
        lang_suffix = lang2suffix[lang]
        for repo in tqdm(repos):
            if repo.get("content"):
                print(f"Skipping {repo['repo']} as it already has content.")
                continue

            repo_name = repo["repo"]
            commit_sha = repo["commit_sha"]
            entrypoint = repo["entrypoint_path"]
            query = f"repo:{repo_name}"

            gh_repos = g.search_repositories(query)
            gh_repo: Repository = gh_repos[0]
            contents = [
                item
                for item in gh_repo.get_contents(entrypoint, ref=commit_sha)
                if any([item.path.endswith(suffix) for suffix in lang_suffix])
            ]

            repo["content"] = {}
            for item in contents:
                if item.encoding != "base64":
                    continue
                file_content = item.decoded_content.decode("utf-8")
                repo["content"][item.path] = file_content

    with open(target_path, "w") as f_out:
        json.dump(lists, f_out)


if __name__ == "__main__":
    Fire(main)
