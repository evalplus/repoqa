# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Dict, List, Optional

import openai

from repoqa.provider.request.openai import make_auto_request

ISSUE_SEP = (
    "-" * 50
    + "\nThe below is the discussion and comments on the question:"
    + "\n"
    + "-" * 50
)
GEN_QUESTION_ANS_SEP = "\n==========\n"


def extract_answers(jsonl_file: Path, key: str) -> Dict[str, Dict[str, str]]:
    assert key in [
        "issue_id",
        "question_id",
    ], "Key must be either 'issue_id' or 'question_id'"
    answers = {}
    with open(jsonl_file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            answers[data[key]] = data
    return answers


def retrieve_code_context_files(
    dataset_path: str,
    issue_content: str,
    repo_name: str,
    relevant_file_paths: Optional[List[str]] = None,
) -> Dict[str, str]:
    with open(dataset_path, "r") as f:
        lists = json.load(f)

    for lang, repos in lists.items():
        for repo in repos:
            if repo["repo"] == repo_name:
                repo_content = repo["content"]  # dict of {file_path: code}
                if relevant_file_paths is not None:
                    repo_content_relevant = {
                        file_path: repo_content[file_path]
                        for file_path in relevant_file_paths
                    }
                else:
                    relevant_file_paths = get_potential_context_files(
                        repo_content, repo_name, issue_content
                    )
                    repo_content_relevant = {
                        file_path: repo_content[file_path]
                        for file_path in relevant_file_paths
                    }
                return repo_content_relevant

    raise ValueError(f"Repository {repo_name} not found in the dataset")


def truncate_context_files_if_too_large(
    issue_or_question_id: str, code_context_dict: Dict[str, str], max_lines: int = 2000
) -> Dict[str, str]:
    # sort the code context by lines of code from smallest to largest
    code_context_dict = dict(
        sorted(code_context_dict.items(), key=lambda x: x[1].count("\n"))
    )
    code_context = f"\n\n".join(
        [
            f"File: {file_path}\n\n{code}"
            for file_path, code in code_context_dict.items()
        ]
    )
    if code_context.count("\n") > max_lines:
        # span the context files to the first max_lines lines
        code_context = ""
        for idx, (file_path, code) in enumerate(code_context_dict.items()):
            if code_context.count("\n") + code.count("\n") > max_lines:
                print(
                    f"[WARNING] Code context of issue or question {issue_or_question_id} is too large, limiting to {idx} files"
                )
                break
            code_context += f"File: {file_path}\n\n{code}\n\n"
    return code_context


def get_potential_context_files(
    repo_content: Dict[str, str], repo_name: str, issue_content: str
) -> List[str]:
    # use OpenAI GPT-4 to decide which code context is relevant to the issue
    client = openai.Client()
    file_list = "\n".join([f"{file_path}" for file_path in repo_content.keys()])
    prompt = f"Here is a real-world GitHub issue from the repository {repo_name}:\n\n{issue_content}\n\nThe below is a list of all code files in the repository:\n\n{file_list}\n\nPlease select up to 10 code files that may be relevant to the issue above.\n\nPlease return the file paths in a list split by ', ' like 'path/to/file1.py, path/to/file2.py, path/to/file3.py'.\n\n Do not reply anything else other than the file paths."

    output = make_auto_request(
        client, prompt, "gpt-4o", max_tokens=1000, temperature=0, n=1
    )
    relevant_file_paths = output.choices[0].message.content.split(", ")
    for path in relevant_file_paths:
        assert (
            path in repo_content.keys()
        ), f"File path {path} is not in the repository content"
    return relevant_file_paths


def get_code_context_from_gen_question_jsonl(
    gen_question_jsonl_file: Path, repo_name: str, middle_func_name: str
) -> str:
    with open(gen_question_jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["repo"] == repo_name and data["name"] == middle_func_name:
                return data["code"]
    raise ValueError(
        f"Function {middle_func_name} not found in the generated question JSONL file"
    )
