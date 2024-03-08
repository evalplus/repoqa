# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from fire import Fire
from github import Auth, Github
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
    language: str = "python",
    stars: int = 100,
    minimal_new_commits: int = 50,
    new_commit_since: str = "2024-01-01",
    minimal_lang_bytes: int = 1024 * 64,  # 64k ideally
):
    token = os.getenv("GITHUB_TOKEN")
    assert token is not None, "Make a token at https://github.com/settings/tokens"
    auth = Auth.Token(token)

    # See repo selection criteria at https://github.com/evalplus/repoqa/issues/1
    query = []
    query.append(f"language:{language}")
    query.append(f"stars:>={stars}")
    query.append("license:mit license:apache-2.0")
    query.append(f"pushed:>={new_commit_since}")
    # 128KB to 32MB
    query.append("size:128..32000")
    query.append("sort:stars")

    # compile query
    query = " ".join(query)
    print(f"{query=}")
    g = Github(auth=auth, per_page=100)

    lang_suffix = lang2suffix[language]

    with open(f"{language}-{datetime.now().isoformat()}.jsonl", "w") as f_out:
        repos = g.search_repositories(query)
        print("Found ", repos.totalCount, "repositories for", language)
        for repo in tqdm(repos, total=repos.totalCount):
            # filter at least 100 commits have been made since 2023 Q4 (>= 2023-09-01).
            commits = repo.get_commits()
            if (
                count_ := sum(
                    True
                    for commit in commits
                    if commit.last_modified_datetime
                    >= datetime.strptime(new_commit_since, "%Y-%m-%d").replace(
                        tzinfo=ZoneInfo("UTC")
                    )
                )
            ) < minimal_new_commits:
                print(
                    f"Skip {repo.html_url} for have less than {minimal_new_commits} after {new_commit_since} (only {count_} commits)"
                )
                continue

            # filter repos that is large enough
            git_tree = repo.get_git_tree(repo.default_branch, recursive=True)

            tree_iter = list(
                filter(
                    lambda item: item.type == "blob"
                    and any([item.path.endswith(suffix) for suffix in lang_suffix]),
                    tqdm(git_tree.tree, leave=False),
                )
            )

            code_file_size = int(sum(item.size for item in tree_iter))
            if code_file_size < minimal_lang_bytes:
                print(
                    f"Skip {repo.html_url} for have less than {minimal_lang_bytes} bytes source file after {new_commit_since} (only {code_file_size} bytes)"
                )
                continue

            schema = dict(
                repo_name=repo.name,
                repo_size=code_file_size,
                repo_owner=repo.owner.login,
                repo_url=repo.html_url,
                commit_sha=git_tree.sha,
                last_modified_time=git_tree.last_modified_datetime,
                content={},
            )
            for item in tree_iter:
                # Fetch the content for each Python file
                content = repo.get_contents(item.path)
                assert not isinstance(content, list)
                if content.encoding != "base64":
                    continue
                file_content = content.decoded_content.decode("utf-8")
                schema["content"][item.path] = file_content
            f_out.write(json.dumps(schema) + "\n")
            f_out.flush()


if __name__ == "__main__":
    Fire(main)
