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
    GEN_QUESTION_ANS_SEP,
    retrieve_code_context_files,
    truncate_context_files_if_too_large,
)


def question_answer_gen(
    repo: str,
    question_content: str,
    model: str,
    code_context: Optional[str] = None,
    base_url: str = None,
    backend: str = "openai",
    max_new_tokens: int = 2048,
) -> str:
    if backend == "openai":
        client = openai.Client()
    else:
        raise NotImplementedError("Only openai is supported for now")

    prompt = f"Here is a question on the repository {repo}:\n\n{question_content}\n\nPlease provide a brief answer to the issue above.\n\n"
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
    gen_question_jsonl_file: str,
    model: str = "gpt-4o",
    output_path: str = "gen_question_answer.jsonl",
    use_batch_api: bool = False,
):
    assert use_batch_api == False, "Batch API is not supported yet."
    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    assert os.path.exists(
        gen_question_jsonl_file
    ), "Generated question JSONL file does not exist"

    with open(output_path, "+a") as f_out:
        with open(gen_question_jsonl_file, "r") as f:
            for line in f:
                data = json.loads(line)
                repo = data["repo"]
                mid_func_name = data["name"]
                code_context = data["code"]
                response = data["response"]
                elements = response.split(GEN_QUESTION_ANS_SEP)
                for element in elements:
                    if element.strip() == "":
                        continue
                    # E.g. **Question_1**: What is the primary purpose of the tool in this repository?\n**Answer_1**: The tool is designed as an uncompromising code formatter for Python, aiming to standardize the formatting of Python code across projects.
                    question_id = (
                        element.split("\n")[0].split(":")[0].strip().replace("**", "")
                        + "#"
                        + mid_func_name
                    )
                    question = element.split("\n")[0].split(":")[1].strip()
                    gt_answer = element.split("\n")[1].split(":")[1].strip()

                    gen_question_answer_no_context = question_answer_gen(
                        repo, question, model, backend="openai"
                    )
                    gen_question_answer_with_context = question_answer_gen(
                        repo, question, model, code_context, backend="openai"
                    )

                    result = {
                        "repo": repo,
                        "question_id": question_id,
                        "question": question,
                        "gt_answer": gt_answer,
                        "model": model,
                        "answer_no_context": gen_question_answer_no_context,
                        "answer_with_context": gen_question_answer_with_context,
                    }
                    json.dump(result, f_out)
                    f_out.write("\n")
                    f_out.flush()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
