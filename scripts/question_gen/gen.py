# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
from pathlib import Path
from typing import List

import openai
from tqdm import tqdm

from repoqa.provider.request.openai import make_auto_request
from repoqa.utility import topological_sort

CAPTURE_HEAD = "<desc_start>"
CAPTURE_TAIL = "<desc_end>"

NUM_QUESTIONS = 5


def get_few_shot_examples(few_shot_example_dir: Path) -> List[str]:
    return [
        example_txt.read_text().rstrip()
        for example_txt in few_shot_example_dir.glob("*.txt")
    ]


def make_prompt(repo_name: str, code: str, few_shot_examples: List[str] = None) -> str:
    instruction = f'Imagine you are a developer who is new to the repository, and you may have some questions regarding the repo. Can you ask {NUM_QUESTIONS} factual questions regarding the repo "{repo_name}" below and provide **brief** answers correspondingly?'
    few_shot_instruction = "Here are some examples of questions and answers mined from real-world GitHub issues that you can learn from:"
    if len(few_shot_examples) > 0:
        few_shot_prompt = (
            few_shot_instruction + "\n\n" + "\n\n".join(few_shot_examples) + "\n\n"
        )
    else:
        few_shot_prompt = ""
    return f"""\
{instruction}

```
{code}
```

{instruction}

{few_shot_prompt}

Please follow format to complete the skeleton below:

{CAPTURE_HEAD}
==========
**Question_1**: ...
**Answer_1**: ...
==========
**Question_2**: ...
**Answer_2**: ...
==========
**Question_3**: ...
**Answer_3**: ...
==========
...
{CAPTURE_TAIL}

{instruction}

Notes:
1. DO NOT reveal function names ({repo_name}) and variable names
2. Start with {CAPTURE_HEAD} and end with {CAPTURE_TAIL}
3. Customize the description to differentiate it from other functions
"""


# Question generation from given repo code snippets
def main(
    dataset_path: str,
    code_ctx_lines: int = 1000,
    output_desc_path: str = "question_generation.jsonl",
    use_batch_api: bool = False,
    verbose: bool = False,
    debug: bool = False,
    num_fewshots: int = 0,  # 0 for zero-shot, otherwise few-shot
):
    assert use_batch_api == False, "Batch API is not supported yet."

    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    with open(dataset_path, "r") as f:
        lists = json.load(f)

    # resume from output_desc_path
    if output_desc_path.endswith(".jsonl") and os.path.exists(output_desc_path):
        with open(output_desc_path, "r") as f:
            results = [json.loads(line) for line in f]
    else:
        # {repo, name, prompt, response}
        results = []

    # a set of inference task to run; each item is a tuple of {repo, name, prompt}
    tasks = []
    for lang, repos in lists.items():
        print(f"🔥 Collecting unannotated needle functions for {lang}")
        for repo in tqdm(repos):
            if not repo.get("dependency"):
                print(
                    f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `dependency` -- do dependency analysis first"
                )
                continue
            ordered_paths = topological_sort(repo["dependency"])
            repo_lines = []
            for path in ordered_paths:
                repo_lines.extend(repo["content"][path].split("\n"))

            def get_code(global_start_line, global_end_line):
                return "\n".join(
                    repo_lines[
                        max(0, global_start_line - code_ctx_lines) : min(
                            global_end_line + code_ctx_lines, len(repo_lines)
                        )
                    ]
                )

            existing_needles = set(
                [item["name"] for item in results if item["repo"] == repo["repo"]]
            )

            for needle in repo["needles"][:1]:
                needle_fn_name = needle[
                    "name"
                ]  # the function in the middle of the context
                if needle_fn_name in existing_needles:
                    continue
                code = get_code(
                    needle["global_start_line"], needle["global_end_line"]
                )  # to be fixed
                print("*" * 80)
                print(code)
                print("*" * 80)
                all_few_shot_examples = get_few_shot_examples(
                    Path(__file__).parent / "few_shot_examples"
                )

                random.seed(42)
                if num_fewshots > 0:
                    few_shot_examples = random.sample(
                        all_few_shot_examples, num_fewshots
                    )
                else:
                    few_shot_examples = []
                prompt = make_prompt(repo["repo"], code, few_shot_examples)
                if verbose:
                    print(prompt)
                    print("-" * 80)
                tasks.append(
                    {
                        "repo": repo["repo"],
                        "name": needle_fn_name,
                        "prompt": prompt,
                        "code": code,
                    }
                )

    print(f"🔥 {len(tasks)} needle functions to be annotated in total")
    client = openai.Client()
    with open(output_desc_path, "+a") as f_out:
        for task in tqdm(tasks):
            print(f"🔥 Annotating {task['name']} in {task['repo']}")
            output = make_auto_request(
                client,
                task["prompt"],
                model="gpt-4-turbo",
                max_tokens=2048,
                temperature=0.2,
                n=1,
            )
            raw_response = output.choices[0].message.content
            result = {
                "repo": task["repo"],
                "name": task["name"],
                "prompt": task["prompt"],
                "middle_function_name": task["name"],
                "code": task["code"],
                "raw_response": raw_response,
                "response": raw_response.split(CAPTURE_HEAD)[-1].split(CAPTURE_TAIL)[0],
            }
            json.dump(result, f_out)
            f_out.write("\n")
            f_out.flush()

            if debug:
                print("[PROMPT]", "-" * 80)
                # the prompt is too long, so we print the last 200 lines
                print("\n".join(task["prompt"].split("\n")[-200:]))
                print("[RESPONSE]", "-" * 80)
                print(raw_response)
                print("-" * 80)
                print("Enter to continue... or b to break:")
                if input() == "b":
                    break


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
