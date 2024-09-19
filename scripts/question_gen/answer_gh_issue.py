# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Optional

import openai

from repoqa.provider.request.openai import make_auto_request
from scripts.question_gen import (
    ISSUE_SEP,
    retrieve_code_context_files,
    truncate_context_files_if_too_large,
)


def strip_issue_question(issue_content: str) -> str:
    issue_question_content = issue_content.split(ISSUE_SEP)[0].strip()
    issue_replies = issue_content.split(ISSUE_SEP)[1].strip()

    # 0-idx is the first reply
    if "(QUESTIONER) replies:" in issue_replies.split("\n")[0]:
        for (idx, line) in enumerate(issue_replies.split("\n")):
            if "replies: " in line and idx > 0:
                break
        issue_self_reply = "\n".join(issue_replies.split("\n")[:idx])
        issue_question_content = f"{issue_question_content}\n\n{issue_self_reply}"

    return issue_question_content


def issue_answer_gen(
    repo: str,
    issue_content: str,
    model: str,
    code_context: Optional[str] = None,
    base_url: str = None,
    backend: str = "openai",
    max_new_tokens: int = 2048,
) -> str:
    issue_question_content = strip_issue_question(issue_content)
    if backend == "openai":
        client = openai.Client()
    else:
        raise NotImplementedError("Only openai is supported for now")

    prompt = f"Here is a real-world GitHub issue from the repository {repo}:\n\n{issue_question_content}\n\nPlease provide a brief answer to the issue above.\n\n"
    if code_context is not None:
        prompt = f"{prompt}\n\n Here is the code context that may be relevant to this issue:\n\n{code_context}\n\n"
    output = make_auto_request(
        client,
        prompt,
        model,
        max_tokens=max_new_tokens,
        temperature=0.2,
        n=1,
    )

    return output.choices[0].message.content


def main(
    dataset_path: str,
    issue_dir: str,
    max_ctx_lines: int = 2000,
    model: str = "gpt-4o",  # we use the best gpt-4o as ground truth to filter issues
    output_path: str = "gh_issue_answer.jsonl",
    use_batch_api: bool = False,
):
    assert use_batch_api == False, "Batch API is not supported yet."
    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    assert os.path.exists(issue_dir), "Issue directory does not exist"

    with open(output_path, "+a") as f_out:
        for issue_file in Path(issue_dir).glob("*.txt"):
            issue_content = issue_file.read_text()
            issue_file_name = issue_file.stem
            issue_repo_name = "/".join(issue_file_name.split("_")[:2])
            code_context_dict = retrieve_code_context_files(
                dataset_path, issue_content, issue_repo_name
            )
            limitted_code_context = truncate_context_files_if_too_large(
                issue_file_name, code_context_dict, max_ctx_lines
            )
            issue_answer_no_context = issue_answer_gen(
                issue_repo_name, issue_content, model
            )
            issue_answer_with_context = issue_answer_gen(
                issue_repo_name,
                issue_content,
                model,
                code_context=limitted_code_context,
            )

            result = {
                "repo": issue_repo_name,
                "issue_id": issue_file_name.replace(".txt", ""),
                "code_context_files": list(code_context_dict.keys()),
                "answer_no_context": issue_answer_no_context,
                "answer_with_context": issue_answer_with_context,
            }
            json.dump(result, f_out)
            f_out.write("\n")
            f_out.flush()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
