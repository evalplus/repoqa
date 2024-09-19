# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import openai

from repoqa.provider.request.openai import make_auto_request
from scripts.question_gen import (
    extract_answers,
    get_code_context_from_gen_question_jsonl,
    retrieve_code_context_files,
    truncate_context_files_if_too_large,
)


def judge_two_answers(
    repo: str,
    issue_or_question_content: str,
    issue_or_question_id: str,
    ans_1: str,
    ans_2: str,
    model: str,
    code_context: str,
    base_url: str = None,
    gt_answer: str = None,
    backend: str = "openai",
    max_new_tokens: int = 2048,
) -> str:
    if backend == "openai":
        client = openai.Client()
    else:
        raise NotImplementedError("Only openai is supported for now")

    if (
        issue_or_question_id.startswith(repo)
        and issue_or_question_id.replace(repo + "_", "").isdigit()
    ):
        prompt = f"Here is a real-world GitHub issue from the repository {repo} (note that the issue is closed and may contain the answer of the question, but only the question was visible to the answerer):"
    else:
        prompt = f"Here is a question on the repository {repo}:"
    prompt += f"""\n\n{issue_or_question_content}.
    The below are two answers to the issue/question above. Please judge which one is better and give an explanation.
    <ANSWER_1_BEGIN>
    {ans_1}
    <ANSWER_1_END>
    <ANSWER_2_BEGIN>
    {ans_2}
    <ANSWER_2_END>

    Please answer in the following format:
    ANSWER_1 or ANSWER_2
    <EXPLANATION_BEGIN>
    EXPLANATION
    <EXPLANATION_END>
    """

    if code_context is not None:
        prompt = f"{prompt}\n\n Here is the code context that may be relevant to this issue:\n\n{code_context}\n\n"

    if gt_answer is not None:
        prompt = f"{prompt}\n\n Here is the ground truth answer that you should take as the reference:\n\n{gt_answer}\n\nNote that you should prefer the answer that is more similar/consistent to the ground truth answer."

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
    key: str,
    dataset_path: str,
    answer_path: str,  # eg. "gh_issue_answer.jsonl"
    output_path: str,  # eg. "judge_gh_issue_answer.jsonl"
    judge_model: str = "gpt-4o",
    max_ctx_lines: int = 2000,
    use_batch_api: bool = False,
    issue_dir: str = None,
    gen_question_jsonl_file: str = None,
):
    assert use_batch_api == False, "Batch API is not supported yet."
    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    assert key in [
        "issue_id",
        "question_id",
    ], "Key must be either 'issue_id' or 'question_id'"
    if key == "issue_id":
        assert os.path.exists(issue_dir), "Issue directory does not exist"
    else:
        assert os.path.exists(
            gen_question_jsonl_file
        ), "Generated question JSONL file does not exist"

    answers = extract_answers(answer_path, key)
    with open(output_path, "+a") as f_out:
        # for issue_file in Path(issue_dir).glob("*.txt"):
        # for issue_or_question_id in answers.keys():
        for issue_or_question_id in answers.keys():
            repo_name = answers[issue_or_question_id]["repo"]
            if key == "issue_id":
                issue_or_question_file = Path(issue_dir) / f"{issue_or_question_id}.txt"
                issue_or_question_content = issue_or_question_file.read_text()

                code_context_dict = retrieve_code_context_files(
                    dataset_path,
                    issue_or_question_content,
                    repo_name,
                    answers[issue_or_question_id]["code_context_files"],
                )
                # cut the context files if too large
                code_context = truncate_context_files_if_too_large(
                    issue_or_question_id, code_context_dict, max_ctx_lines
                )
            else:
                issue_or_question_content = answers[issue_or_question_id]["question"]
                middle_func_name = issue_or_question_id.split("#")[1]
                code_context = get_code_context_from_gen_question_jsonl(
                    gen_question_jsonl_file, repo_name, middle_func_name
                )

            issue_answer_no_context = answers[issue_or_question_id]["answer_no_context"]
            issue_answer_with_context = answers[issue_or_question_id][
                "answer_with_context"
            ]
            judge_result = judge_two_answers(
                repo_name,
                issue_or_question_content,
                issue_or_question_id,
                issue_answer_no_context,
                issue_answer_with_context,
                judge_model,
                code_context,
            )

            better_answer_id = judge_result.split("<EXPLANATION_BEGIN>")[0].strip()
            explanation = (
                judge_result.split("<EXPLANATION_BEGIN>")[1]
                .split("<EXPLANATION_END>")[0]
                .strip()
            )
            better_answer = (
                "no_context" if better_answer_id == "ANSWER_1" else "with_context"
            )

            result = {
                "repo": repo_name,
                "issue_or_question_id": issue_or_question_id,
                "better_answer": better_answer,
                "explanation": explanation,
            }
            json.dump(result, f_out)
            f_out.write("\n")
            f_out.flush()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
