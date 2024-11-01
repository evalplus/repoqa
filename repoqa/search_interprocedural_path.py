# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import os
import types
from typing import List, Tuple

from transformers import AutoTokenizer

from repoqa.utility import COMMENT_PREFIX, progress

# Maximum number of test cases from each repo
NUM_TESTCASES = 15

CAPTURE_HEAD = "<path_start>"
CAPTURE_TAIL = "<path_end>"

INSTRUCTION = (
    "Given the following code files, a starting function and an ending function,"
    " please find the shortest path of function calls begining from the starting function"
    " and finishing at the ending function. The analysis should be context insensitive. In"
    " other words, only answer based on the function call declaration and do not "
    " consider any execution."
)

OUTPUT_FORMAT = f"""Please output your response in the following block, seperated by commas:
{CAPTURE_HEAD}
starting_function, function_one, ..., ending_function
{CAPTURE_TAIL}
For each function please follow the naming practice: <FilePath>::<ClassName>.method_name, if \
there is no class for the function, use <FilePath>::method.
"""


def make_task_id(lang, repo, start, end):
    return f"{lang}::{repo}::{start}::{end}"


# TODO: code2flow has an issue with build in methods,
# filtering out faulty paths here, move to scripts/curate
def check_faulty_case(path: List[str]) -> bool:
    def is_builtin_name(name):
        if name in dir(builtins):
            return True
        for type_name in dir(builtins):
            obj = getattr(builtins, type_name)
            if isinstance(obj, type) and name in dir(obj):
                return True
        return False

    for function in path:
        if "." in function:
            last_part = function.split(".")[-1]
        else:
            last_part = function.split("::")[-1]
        if is_builtin_name(last_part):
            return True
    return False


def prepare_code_context(
    file_name_list: List[str], content_mapping, language: str
) -> Tuple[str, int]:
    """
    Much more simplified version from search_needle_function.py.
    Instead of gettting context tokens, we simply modify the file
    content list to ensure that the file name is at the top of each
    file and concatenate them together
    """
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    output_context = ""
    for file_name in file_name_list:
        output_context += f"{COMMENT_PREFIX[language]} File Path: {file_name}\n"
        output_context += content_mapping[file_name] + "\n"

    return (output_context, len(tokenizer.tokenize(output_context)))


def main(
    model: str,
    dataset_path: str,
    base_url: str = None,
    backend: str = None,
    tensor_parallel_size: int = 1,
    code_context_size: int = 16 * 1024,
    max_new_tokens: int = 1024,
    result_dir: str = "results",
    trust_remote_code: bool = False,
    attn_implementation=None,
):
    if backend is None:
        if base_url is not None:
            backend = "openai"
        else:
            backend = "vllm"
        print(f"Using {backend} as the backend")
    assert backend is not None, "Please specify the backend"

    if dataset_path is not None:
        with open(dataset_path) as f:
            dataset = json.load(f)
    else:
        dataset = get_repoqa_data()

    # makedir if not exists
    os.makedirs(result_dir, exist_ok=True)
    context_size_dir = os.path.join(result_dir, f"ntoken_{code_context_size}")
    os.makedirs(context_size_dir, exist_ok=True)
    model_output_path = os.path.join(
        context_size_dir,
        f"{model.replace('/', '_slash_')}-interprocedural.jsonl",
    )

    # resume from model_output_file
    if os.path.exists(model_output_path):
        with open(model_output_path) as f:
            model_outputs = [json.loads(line) for line in f]
    else:
        model_outputs = []

    resumed_task_ids = {
        make_task_id(r["language"], r["repo"], r["name"]) for r in model_outputs
    }

    with open(dataset_path) as f:
        dataset = json.load(f)

    tasks = []
    # for each task we include
    # "repo", "start_function", "end_function", "path", "files"
    for lang, repos in dataset.items():
        # TODO: remove this
        if lang != "python":
            continue

        with progress(f"Processing {lang} context") as pbar:
            for repo in pbar.track(repos):
                # skip if the repo does not have interprocedural
                if "interprocedural_pairs" not in repo:
                    pbar.console.print(
                        f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `interprocedural` -- fetch interprocedural pairs first"
                    )
                    continue

                # Select first ten test cases
                target_testcases = min(
                    NUM_TESTCASES, len(repo["interprocedural_pairs"])
                )

                for interprocedural in repo["interprocedural_pairs"]:

                    task_id = make_task_id(
                        lang,
                        repo["repo"],
                        interprocedural["start"],
                        interprocedural["end"],
                    )
                    if task_id in resumed_task_ids:
                        pbar.console.print(
                            f"Skipping {task_id} as it is already in the results"
                        )
                        continue

                    if check_faulty_case(interprocedural["path"]):
                        continue
                    file_list = interprocedural["files"]
                    task = {
                        "language": lang,
                        "repo": repo["repo"],
                        "start": interprocedural["start"],
                        "end": interprocedural["end"],
                        "groundtruth_path": interprocedural["path"],
                        "context": prepare_code_context(
                            file_list, repo["content"], "python"
                        ),
                    }
                    prompt = INSTRUCTION + OUTPUT_FORMAT + task["context"][0]
                    prompt += f"START FUNCTION: {task['start']}"
                    prompt += f"END FUNCTION: {task['end']}"
                    prompt += INSTRUCTION + OUTPUT_FORMAT
                    task["prompt"] = prompt
                    tasks.append(task)
                    if not (target_testcases):
                        break
                    target_testcases -= 1

    if len(tasks) == 0:
        print("No tasks to evaluate! Exiting...")
        return

    if backend == "openai":
        from repoqa.provider.openai import OpenAIProvider

        engine = OpenAIProvider(model, base_url=base_url)
    elif backend == "vllm":
        from repoqa.provider.vllm import VllmProvider

        engine = VllmProvider(
            model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=int(code_context_size * 1.5),  # Magic number
            trust_remote_code=trust_remote_code,
        )
    elif backend == "anthropic":
        from repoqa.provider.anthropic import AnthropicProvider

        engine = AnthropicProvider(model)
    elif backend == "hf":
        from repoqa.provider.hf import HfProvider

        engine = HfProvider(
            model,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
    elif backend == "google":
        from repoqa.provider.google import GoogleProvider

        engine = GoogleProvider(model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    with open(model_output_path, "a") as f_out:
        with progress(f"Running {model}") as pbar:
            for task in pbar.track(tasks):
                replies = engine.generate_reply(
                    task["prompt"], n=1, max_tokens=max_new_tokens
                )
                print(replies)
                result = {**task, "output": replies}
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                model_outputs.append(result)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
