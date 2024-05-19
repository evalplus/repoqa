# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Tuple

from transformers import AutoTokenizer

from repoqa.compute_score import compute_score, save_json
from repoqa.data import CACHE_DIR, get_repoqa_data
from repoqa.utility import progress, topological_sort

COMMENT_PREFIX = {
    "python": "#",
    "java": "//",
    "typescript": "//",
    "rust": "//",
    "cpp": "//",
}


# Model context below:
TEMPLATE = "instruction\ncode_context\ndescription\ninstruction"

INSTRUCTION = (
    "Based on the function description and code context,"
    " please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:"
)


# Count number of tokens ignoring null byte (used for obfuscate-nl)
def count_tokens(tokens: List[str]) -> int:
    return len([x for x in tokens if x != "<0x00>"])


def _backward_tokenizable_lines(lines, tokenizer, max_tokens):
    """Return the text and tokens from bottom to top"""
    text = ""
    ntokens = 0
    is_break = False
    for line in reversed(lines):
        new_ntokens = count_tokens(tokenizer.tokenize(line + "\n"))
        if ntokens + new_ntokens > max_tokens:
            is_break = True
            break
        text = line + "\n" + text
        ntokens += new_ntokens
    return text, ntokens, is_break


def _forward_tokenizable_lines(lines, tokenizer, max_tokens):
    """Return the text and tokens from top to bottom"""
    text = ""
    ntokens = 0
    is_break = False
    for line in lines:
        new_ntokens = count_tokens(tokenizer.tokenize(line + "\n"))
        if ntokens + new_ntokens > max_tokens:
            is_break = True
            break
        text += line + "\n"
        ntokens += new_ntokens
    if is_break:
        text = text + "...\n"
        ntokens += count_tokens(tokenizer.tokenize("...\n"))
    return text, ntokens, is_break


def make_code_context(
    needle,
    file_content_list: List[Tuple[str, str]],
    position_ratio,
    code_context_size,
    language,
) -> str:
    """
    Slice the file_content_list such that:
    1. The slice contains code_context_size tokens
    2. The positon of the needle is at position_ratio of the slice*
    *May not be achievable if the needle is too close to the beginning or end of the file_content_list
    *May not be accurate as we will also insert file names at the beginning of each file
    *Token sizes might not be 100 accurate but should be close enough
    """
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    needle_file_idx, needle_file_content = [
        (i, content)
        for i, (f, content) in enumerate(file_content_list)
        if f == needle["path"]
    ][0]

    needle_code = needle_file_content[needle["start_byte"] : needle["end_byte"]]
    ntoken_needle = count_tokens(tokenizer.tokenize(needle_code))

    prefix_size = int(code_context_size * position_ratio - ntoken_needle / 2)
    suffix_size = code_context_size - ntoken_needle - prefix_size

    # handling prefix of the needle file
    code_prefix, ntokens, is_break = _backward_tokenizable_lines(
        [COMMENT_PREFIX[language] + " Path: " + needle["path"]]
        + needle_file_content[: needle["start_byte"]].split("\n"),
        tokenizer,
        prefix_size,
    )
    prefix_size -= ntokens

    # handling prefix of the previous files
    index = needle_file_idx - 1
    while not is_break and prefix_size > 0 and index >= 0:
        path, content = file_content_list[index]
        prefix, ntokens, is_break = _forward_tokenizable_lines(
            [COMMENT_PREFIX[language] + " Path: " + path] + content.split("\n"),
            tokenizer,
            prefix_size,
        )
        code_prefix = prefix + code_prefix
        prefix_size -= ntokens
        index -= 1

    # handling suffix of the needle file
    code_suffix, ntokens, is_break = _forward_tokenizable_lines(
        needle_file_content[needle["end_byte"] :].split("\n"), tokenizer, suffix_size
    )
    suffix_size -= ntokens

    # handling suffix of the next files
    index = needle_file_idx + 1
    while not is_break and suffix_size > 0 and index < len(file_content_list):
        path, content = file_content_list[index]
        suffix, ntokens, is_break = _forward_tokenizable_lines(
            [COMMENT_PREFIX[language] + " Path: " + path] + content.split("\n"),
            tokenizer,
            suffix_size,
        )
        code_suffix += suffix
        suffix_size -= ntokens
        index += 1

    # Remove all null characters from all three portions (obfuscate-nl)
    code_prefix = code_prefix.replace("\0", "")
    needle_code = needle_code.replace("\0", "")
    code_suffix = code_suffix.replace("\0", "")

    code_context = code_prefix + needle_code + code_suffix

    needle_token_start = len(tokenizer.tokenize(code_prefix))
    needle_token_end = needle_token_start + len(tokenizer.tokenize(needle_code))
    code_context_ntokens = needle_token_end + len(tokenizer.tokenize(code_suffix))

    return {
        "code_context": code_context,
        "needle_token_start": needle_token_start,
        "needle_token_end": needle_token_end,
        "code_context_ntokens": code_context_ntokens,
    }


def make_task_id(lang, repo, needle_name):
    return f"{lang}::{repo}::{needle_name}"


def make_cache_id(lang, repo, needle_name, code_context_size, position_ratio):
    return f"{lang}::{repo}::{needle_name}::{code_context_size}::{position_ratio}"


def evaluate_model(
    model: str,
    base_url: str = None,
    backend: str = None,
    tensor_parallel_size: int = 1,
    code_context_size: int = 16 * 1024,
    max_new_tokens: int = 1024,
    result_dir: str = "results",
    languages: List[str] = None,
    caching: bool = True,  # if enabled, will cache the tasks which can be used to resume
    system_message: str = None,
    dataset_path: str = None,
    ignore_comments: bool = False,
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
        f"{model.replace('/', '_slash_')}.jsonl",
    )

    # resume from model_output_file
    if os.path.exists(model_output_path):
        with open(model_output_path) as f:
            model_outputs = [json.loads(line) for line in f]
    else:
        model_outputs = []

    # resume tasks from cache if any
    # schema: {"cache_id": .., **task}
    cache_file = os.path.join(CACHE_DIR, f"cache_ntoken_{code_context_size}_v1.jsonl")
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache = {}
    if caching:
        print("🔥 Caching enabled")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cache = [json.loads(line) for line in f]
                cache = {c["cache_id"]: c for c in cache}
                # remove the cache_id field in c
                for c in cache.values():
                    c.pop("cache_id")
            print(f"Resuming from cache: {cache_file} with {len(cache)} tasks")

    resumed_task_ids = {
        make_task_id(r["language"], r["repo"], r["name"]) for r in model_outputs
    }

    # for each task we include
    # "repo", "name", "language", "path",
    # "template", "position_ratio", "description", "instruction", "code_context"
    # "needle_token_start", "needle_token_end", "code_context_ntokens"
    tasks = []
    for lang, repos in dataset.items():
        if languages is not None and lang not in languages:
            print(f"Skipping {lang} as it is not selected; selected: {languages}")
            continue

        print(f"🔥 Preparing code context for {lang}...")
        with progress(f"Processing {lang} context") as pbar:
            for repo in pbar.track(repos):
                # skip if the repo does not have needles
                if "needles" not in repo:
                    pbar.console.print(
                        f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `needles` -- do needle analysis first"
                    )
                    continue

                ordered_paths = topological_sort(repo["dependency"])
                file_content_list = [
                    (path, repo["content"][path]) for path in ordered_paths
                ]
                for i, needle in enumerate(repo["needles"]):
                    task_id = make_task_id(lang, repo["repo"], needle["name"])
                    if task_id in resumed_task_ids:
                        pbar.console.print(
                            f"Skipping {task_id} as it is already in the results"
                        )
                        continue

                    position_ratio = (i + 0.5) / len(repo["needles"])
                    cache_id = make_cache_id(
                        lang,
                        repo["repo"],
                        needle["name"],
                        code_context_size,
                        position_ratio,
                    )
                    if cache_id in cache:
                        tasks.append(cache[cache_id])
                        continue

                    task = {
                        "repo": repo["repo"],
                        "name": needle["name"],
                        "language": lang,
                        "path": needle["path"],
                        "position_ratio": position_ratio,
                        "description": f"\nFunction Description:{needle['description']}\n",
                        "instruction": INSTRUCTION,
                        "template": TEMPLATE,
                    }
                    code_context_info = make_code_context(
                        needle,
                        file_content_list,
                        position_ratio=position_ratio,
                        code_context_size=code_context_size,
                        language=lang,
                    )
                    task.update(code_context_info)
                    tasks.append(task)

                    if caching:  # cache
                        with open(cache_file, "a") as f_out:
                            f_out.write(
                                json.dumps({"cache_id": cache_id, **task}) + "\n"
                            )

    # filter finished tasks again (in case a cache is used)
    tasks = [
        task
        for task in tasks
        if make_task_id(task["language"], task["repo"], task["name"])
        not in resumed_task_ids
    ]

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
            max_model_len=int(code_context_size * 1.25),  # Magic number
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

    if not system_message:
        print("🔥 System message is disabled")
        system_message = None

    with open(model_output_path, "a") as f_out:
        with progress(f"Running {model}") as pbar:
            for task in pbar.track(tasks):
                actual_position_ratio = (
                    task["needle_token_start"] / task["code_context_ntokens"]
                )
                pbar.console.print(
                    f"Searching {task['name']} in {task['repo']} ({task['language']}) -- "
                    f"position ratio: actual={actual_position_ratio:.2f}, expected={task['position_ratio']}"
                )
                prompt = ""
                for key in task["template"].split("\n"):
                    prompt += task[key]

                replies = engine.generate_reply(
                    prompt, n=1, max_tokens=max_new_tokens, system_msg=system_message
                )
                result = {**task, "output": replies}
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                model_outputs.append(result)

    file_base, _ = os.path.splitext(model_output_path)
    result_path = file_base + "-SCORES.json"
    output_json = compute_score(model, dataset, model_outputs, ignore_comments)
    save_json(output_json, result_path)


def main():
    from fire import Fire

    Fire(evaluate_model)


if __name__ == "__main__":
    main()
