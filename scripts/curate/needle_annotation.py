# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import openai
from tqdm import tqdm

from repoqa.provider.request.openai import make_auto_request
from repoqa.utility import topological_sort

CAPTURE_HEAD = "<desc_start>"
CAPTURE_TAIL = "<desc_end>"


def make_prompt(fn_name: str, code: str):
    instruction = f'Can you **briefly** describe the purpose, input, output, and procedure of "{fn_name}"?'
    return f"""\
{instruction}

```
{code}
```

{instruction}

Please follow format to complete the skeleton below:

{CAPTURE_HEAD}
1. **Purpose**: ...
2. **Input**: ...
3. **Output**: ...
4. **Procedure**: ...
{CAPTURE_TAIL}

{instruction}

Notes:
1. DO NOT reveal function names ({fn_name}) and variable names
2. Start with {CAPTURE_HEAD} and end with {CAPTURE_TAIL}
3. Customize the description to differentiate it from other functions
"""


# Annotate an incomplete repoqa dataset with function and class information
def main(
    dataset_path: str,
    code_prefix_lines: int = 100,
    output_desc_path: str = "function_description.jsonl",
    use_batch_api: bool = False,
    verbose: bool = False,
    debug: bool = False,
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
        # {repo, name, prompt, annotation}
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
                        max(0, global_start_line - code_prefix_lines) : global_end_line
                    ]
                )

            existing_needles = set(
                [item["name"] for item in results if item["repo"] == repo["repo"]]
            )

            for needle in repo["needles"]:
                fn_name = needle["name"]
                if fn_name in existing_needles:
                    continue
                code = get_code(needle["global_start_line"], needle["global_end_line"])
                prompt = make_prompt(fn_name, code)
                if verbose:
                    print(prompt)
                    print("-" * 80)
                tasks.append(
                    {
                        "repo": repo["repo"],
                        "name": fn_name,
                        "prompt": prompt,
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
            annotation = output.choices[0].message.content
            result = {
                "repo": task["repo"],
                "name": task["name"],
                "prompt": task["prompt"],
                "raw_annotation": annotation,
                "annotation": annotation.split(CAPTURE_HEAD)[-1].split(CAPTURE_TAIL)[0],
            }
            json.dump(result, f_out)
            f_out.write("\n")
            f_out.flush()

            if debug:
                print("[PROMPT]", "-" * 80)
                print(task["prompt"])
                print("[ANNOTATION]", "-" * 80)
                print(annotation)
                print("-" * 80)
                print("Enter to continue... or b to break:")
                if input() == "b":
                    break


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
