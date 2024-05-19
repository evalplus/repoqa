# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tempdir
from rich.console import Console
from rich.table import Table
from transformers import AutoConfig
from tree_sitter_languages import get_language, get_parser

from repoqa.data import get_repoqa_data
from repoqa.metric import compute_function_similarity
from repoqa.utility import COMMENT_QUERY, FUNCTION_QUERY, progress

LANGUAGES = list(FUNCTION_QUERY.keys())
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class Result(Enum):
    BEST_MATCH = "best_match"
    FAIL_MATCH = "fail_match"


# unbiased estimator from https://github.com/openai/human-eval
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def remove_comments(source_code: str, lang: str) -> str:
    source_bytes = bytes(source_code, "utf8")
    parser = get_parser(lang)
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    # Remove comments from source code
    capture_list = []
    for query_str in COMMENT_QUERY[lang]:
        comment_query = get_language(lang).query(query_str)
        capture_list += comment_query.captures(root_node)

    capture_list.sort(key=lambda cap: cap[0].start_byte, reverse=True)

    for node, _ in capture_list:
        source_bytes = source_bytes[: node.start_byte] + source_bytes[node.end_byte :]

    return source_bytes.decode("utf-8")


def sanitize_output(model_output: str, lang: str) -> str:
    model_output = model_output.strip()
    search_pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    code_blocks = re.findall(search_pattern, model_output, re.DOTALL | re.MULTILINE)

    parser = get_parser(lang)
    fn_query = get_language(lang).query(FUNCTION_QUERY[lang])

    # If not code blocks found, simply return model output
    if not code_blocks:
        return model_output

    processed_blocks = []
    for block in code_blocks:
        processed_blocks.append(block)

        # Try to use tree-sitter to parse if possible
        try:
            block_bytes = bytes(block, "utf8")
            tree = parser.parse(block_bytes)
            for capture in fn_query.captures(tree.root_node):
                node, _ = capture
                function_content = block_bytes[node.start_byte : node.end_byte]
                return function_content.decode("utf8")
        except:
            pass

    # no valid functions found by tree-sitter approach return first block
    return processed_blocks[0]


def print_result_table(model_name, pass_results):
    # Printing scores in a table
    table = Table(title=f"Scores (%) of {model_name} at different thresholds")
    table.add_column("Threshold", justify="center", style="bold magenta")
    for threshold in THRESHOLDS:
        table.add_column(f"{threshold}", justify="center")

    # Prepare data to determine the maximum values for each threshold
    threshold_scores = {threshold: [] for threshold in THRESHOLDS}
    for lang_results in pass_results.values():
        for thresh, value in lang_results.items():
            threshold_scores[thresh].append(value["pass@1"])

    # Calculate the maximum score for each threshold
    max_scores = {
        threshold: max(scores) for threshold, scores in threshold_scores.items()
    }
    min_scores = {
        threshold: min(scores) for threshold, scores in threshold_scores.items()
    }

    # Fill the table rows
    for language, lang_results in pass_results.items():
        row = [("⭐" if language == "all" else "") + language]
        for threshold, value in lang_results.items():
            score = value["pass@1"]
            formatted_score = f"{100 * score:.1f}"
            if max_scores[threshold] - score < 0.01:
                formatted_score = f"[bold green]{formatted_score}[/]"
            elif score - min_scores[threshold] < 0.01:
                formatted_score = f"[bold red]{formatted_score}[/]"
            row.append(formatted_score)
        if language == "all":
            row = [f"[bold yellow]{r}[/]" for r in row]
        table.add_row(*row)

    Console().print(table)


def needle_evaluator(
    model_output: str,
    ground_truth: str,
    repo_info: Dict,
    lang: str,
    ignore_comments: bool,
) -> Tuple[Result, str, float]:
    contents = repo_info["content"]
    needles = repo_info["needles"]

    best_target = None
    best_similarity = 0
    sanitized_output = sanitize_output(model_output, lang)
    if ignore_comments:
        sanitized_output = remove_comments(sanitized_output, lang)
    for needle in needles:
        current_path = needle["path"]
        current_name = needle["name"]
        current_func = "\n".join(
            contents[current_path].split("\n")[
                needle["start_line"] : needle["end_line"]
            ]
        )
        if ignore_comments:
            current_func = remove_comments(current_func, lang)

        current_similarity = compute_function_similarity(sanitized_output, current_func)
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_target = current_name

    if best_target == ground_truth:
        verdict = Result.BEST_MATCH
    else:
        verdict = Result.FAIL_MATCH
    return verdict, best_target, best_similarity


def _get_repo(lang_data: Dict, repo_name: str) -> Dict:
    for repo in lang_data:
        if repo["repo"] == repo_name:
            return repo


def compute_language_results(evaluation_result: Dict, all_results: Dict) -> None:
    for language, lang_results in evaluation_result.items():
        current_result = {}
        total = np.array([1 for _ in lang_results])

        for threshold in THRESHOLDS:
            correct_result = []
            for res in lang_results:
                bc = 0
                if res["is_best_similar"] and res["best_similar_score"] >= threshold:
                    bc = 1
                correct_result.append(bc)
            correct_result = np.array(correct_result)

            pass_at_k = {
                f"pass@{k}": estimate_pass_at_k(total, correct_result, k).mean()
                for k in [1, 10, 100]
                if total.min() >= k
            }
            current_result[threshold] = pass_at_k
        all_results[language] = current_result


def fetch_hf_context(model_name: str) -> str:
    # Retrieved from https://github.com/vllm-project/vllm/blob/main/vllm/config.py#L1073
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    try:
        with tempdir.TempDir() as temp_dir:
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=temp_dir,
                force_download=True,
                trust_remote_code=True,
            ).to_dict()
            longest_context = 0
            for key in possible_keys:
                if key in config:
                    longest_context = max(config[key], longest_context)
            if not (longest_context):
                return "N/A"
            return str(int(longest_context / 1024)) + "k"
    except Exception as err:
        print(f"fetching failed... Reason:\n{err}")
        return "N/A"


def compute_score(
    model_name: str, dataset: Dict, model_output: List[Dict], ignore_comments: bool
) -> Dict:
    evaluation_result = defaultdict(list)
    with progress(f"Scoring {model_name}") as pbar:
        for result in pbar.track(model_output):
            lang = result["language"]
            repo_name = result["repo"]
            model_outputs = result["output"]
            ground_truth = result["name"]
            repo_info = _get_repo(dataset[lang], repo_name)

            model_output = model_outputs[0]
            verdict, best_target, best_similarity = needle_evaluator(
                model_output, ground_truth, repo_info, lang, ignore_comments
            )

            is_best_similar = False
            if verdict == Result.BEST_MATCH:
                is_best_similar = True

            current_task = {
                "repo": repo_name,
                "name": ground_truth,
                "needle_position": result["position_ratio"],
                "is_best_similar": is_best_similar,
                "best_similar_score": best_similarity,
                "best_target": best_target,
                "position": {
                    "token_start": result["needle_token_start"],
                    "token_end": result["needle_token_end"],
                },
            }
            evaluation_result[lang].append(current_task)

    # Calculate pass@k
    pass_results = {}

    all_langs = []
    for lang in evaluation_result:
        all_langs += evaluation_result[lang]
    total = np.array([1 for _ in all_langs])

    pass_results["all"] = {}
    for threshold in THRESHOLDS:
        correct_result = []
        for res in all_langs:
            bc = 0
            if res["is_best_similar"] and res["best_similar_score"] >= threshold:
                bc = 1
            correct_result.append(bc)
        correct_result = np.array(correct_result)
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, correct_result, k).mean()
            for k in [1, 10, 100]
            if total.min() >= k
        }
        pass_results["all"][threshold] = pass_at_k

    compute_language_results(evaluation_result, pass_results)
    print_result_table(model_name, pass_results)

    output_json = {}
    model_json = {}
    model_json["eval_date"] = str(datetime.now())

    # hardcode paid models
    if "/" in model_name:
        if model_name.startswith("bigcode/starcoder2"):
            train_context = "16k"
        else:
            train_context = fetch_hf_context(model_name)
    elif model_name.startswith("gpt-4-turbo") or model_name.startswith("gpt-4o-"):
        train_context = "128k"
    elif model_name.startswith("gpt-3.5-"):
        train_context = "16k"
    elif model_name.startswith("gemini-1.5-pro") or model_name.startswith(
        "gemini-1.5-flash"
    ):
        train_context = "1000k"
    elif model_name.startswith("gemini-1.0-pro"):
        train_context = "32k"
    elif model_name.startswith("claude-3-"):
        train_context = "200k"
    else:
        train_context = "N/A"
    model_json["train_size"] = train_context
    model_json["scores"] = pass_results
    model_json["results"] = evaluation_result

    output_json[model_name] = model_json

    return output_json


def get_model_name(output_path: str) -> str:
    file_name = Path(output_path).stem
    segments = file_name.split("_")
    output_name = ""
    for segment in segments:
        if segment == "slash":
            output_name += "/"
        else:
            output_name += segment
    return output_name


def save_json(output_json, result_path) -> None:
    if os.path.isfile(result_path):
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(output_json, f)


def compute_main(
    model_output_path: str, ignore_comments: bool = False, dataset_path: str = None
):
    if dataset_path is None:
        dataset = get_repoqa_data()
    else:
        with open(dataset_path, "r") as dataset_f:
            dataset = json.load(dataset_f)

    model_outputs = []
    with open(model_output_path, "r") as output_f:
        for line in output_f:
            model_outputs.append(json.loads(line))

    file_base, _ = os.path.splitext(model_output_path)
    result_path = file_base + "-SCORES.json"
    model_name = get_model_name(model_output_path)
    output_json = compute_score(model_name, dataset, model_outputs, ignore_comments)
    save_json(output_json, result_path)


def main():
    from fire import Fire

    Fire(compute_main)


if __name__ == "__main__":
    main()
