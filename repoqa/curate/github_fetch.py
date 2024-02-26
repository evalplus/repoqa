# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from fire import Fire
from github import Auth, Github
from tqdm.auto import tqdm


class GitHubRepoMeta(TypedDict):
    repo_name: str
    repo_owner: str
    commit_sha: str
    repo_size: int


class GitHubDocument(GitHubRepoMeta):
    timestamp: str
    path: str
    content: str


@dataclass(frozen=True)
class GitHubFetchingArgs:
    query: str = field(
        default="language:python stars:>=1000 license:mit license:apache-2.0 created:>=2023-05-09",
        metadata={
            "help": "Rule to search for repositories; check https://docs.github.com/en/search-github/searching-on-github/searching-for-repositories"
        },
    )


lang2suffix = {
    "python": [".py"],
    "javascript": [".js"],
    "go": [".go"],
    "c++": [".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx", ".c", ".h"],
    "java": [".java"],
    "typescript": [".ts"],
    "c": [".c", ".h"],
    "c#": [".cs"],
    "php": [".php"],
    "rust": [".rs"],
}


def main(language: str = "python", stars: int = 10):
    token = os.getenv("GITHUB_TOKEN")
    assert token is not None, "Make a token at https://github.com/settings/tokens"
    auth = Auth.Token(token)

    # See repo selection criteria at https://github.com/evalplus/repoqa/issues/1
    query = []
    query.append(f"language:{language}")
    query.append(f"stars:>={stars}")
    query.append("license:mit license:apache-2.0")
    query.append("pushed:>=2023-09-01")
    # 50KB to 10MB
    query.append("size:50..10000")

    # compile query
    query = " ".join(query)
    g = Github(auth=auth, per_page=100)

    lang_suffix = lang2suffix[language]

    repo_sizes = []  # unit of 10KB

    with open(f"{language}-{datetime.now().isoformat()}.jsonl", "w") as f_out:
        repos = g.search_repositories(query)
        print("Total count", repos.totalCount)
        # TODO: apply dependency analysis
        for repo in tqdm(repos, total=repos.totalCount):
            # filter repos diversified over the size, every 10KB
            if repo.size // 10 in repo_sizes:
                continue
            repo_sizes.append(repo.size // 10)

            # filter at least 100 commits have been made since 2023 Q4 (>= 2023-09-01).
            commits = repo.get_commits()
            if (
                sum(
                    True
                    for commit in commits
                    if commit.last_modified_datetime
                    >= datetime(2023, 9, 1, tzinfo=ZoneInfo("UTC"))
                )
                < 100
            ):
                continue

            git_tree = repo.get_git_tree(repo.default_branch, recursive=True)
            tree_iter = filter(
                lambda item: item.type == "blob"
                and any([item.path.endswith(suffix) for suffix in lang_suffix]),
                tqdm(git_tree.tree, leave=False),
            )
            for item in tree_iter:
                # Fetch the content for each Python file
                content = repo.get_contents(item.path)
                assert not isinstance(content, list)
                if content.encoding != "base64":
                    continue
                file_content = content.decoded_content.decode("utf-8")
                timestamp = datetime.now().isoformat()
                data = GitHubDocument(
                    timestamp=timestamp,
                    repo_name=repo.name,
                    repo_owner=repo.owner.login,
                    repo_size=repo.size,
                    commit_sha=git_tree.sha,
                    path=item.path,
                    content=file_content,
                )
                f_out.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    Fire(main)
