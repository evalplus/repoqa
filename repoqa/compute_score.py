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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from function_similarity import compute_function_similarity
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

from repoqa.utility import FUNCTION_QUERY

LANGUAGES = ["cpp", "rust", "java", "python", "typescript"]

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class Result(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


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


def sanitize_output(model_output: str, lang: str) -> str:
    search_pattern = r"```(.*?)```"
    code_blocks = re.findall(search_pattern, model_output, re.DOTALL)

    parser = get_parser(lang)
    fn_query = get_language(lang).query(FUNCTION_QUERY[lang])

    # If not code blocks found, simply return model output
    if not code_blocks:
        return model_output

    processed_blocks = []
    for block in code_blocks:
        # Clean up language specific markdown blocks
        lines = block.split("\n")
        if lines[0] == lang:
            block = "\n".join(lines[1:])

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


def needle_evaluator(
    model_output: str, ground_truth: str, repo_info: Dict, lang: str
) -> Tuple[Result, str, float]:
    contents = repo_info["content"]
    needles = repo_info["needles"]

    best_target = None
    best_similarity = 0
    sanitized_output = sanitize_output(model_output, lang)
    for needle in needles:
        current_path = needle["path"]
        current_name = needle["name"]
        current_func = "\n".join(
            contents[current_path].split("\n")[
                needle["start_line"] : needle["end_line"]
            ]
        )

        current_similarity = compute_function_similarity(sanitized_output, current_func)
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_target = current_name

    if best_target == ground_truth:
        verdict = Result.PASS
    elif best_target == None:
        verdict = Result.ERROR
    else:
        verdict = Result.FAIL
    return verdict, best_target, best_similarity


def get_repo(lang_data: Dict, repo_name: str) -> Dict:
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


def compute_score(
    model_name: str,
    dataset: Dict,
    results: List[Dict],
    output_path: str,
    output_results: bool = True,
) -> None:
    eval_max_tokens = 0
    evaluation_result = defaultdict(list)
    for result in tqdm(results):
        lang = result["language"]
        repo_name = result["repo"]
        model_outputs = result["output"]
        ground_truth = result["name"]
        repo_info = get_repo(dataset[lang], repo_name)

        model_output = model_outputs[0]
        verdict, best_target, best_similarity = needle_evaluator(
            model_output, ground_truth, repo_info, lang
        )

        is_best_similar = False
        if verdict == Result.PASS:
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
        eval_max_tokens = max(eval_max_tokens, result["code_context_ntokens"])
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
    # Printing scores
    for language, lang_results in pass_results.items():
        print(f"Evaluation results across {language}: ")
        for threshold, value in lang_results.items():
            print(f"{threshold}: {value['pass@1']:.3f}", end="\t")
        print("")

    # Saving results
    file_base, file_ext = os.path.splitext(output_path)

    result_path = file_base + "-SCORES" + file_ext

    if not output_results:
        return

    output_json = {}
    model_json = {}
    model_json["eval_date"] = str(datetime.now())
    # TODO Update train and evalsize
    model_json["train_size"] = None
    model_json["eval_size"] = eval_max_tokens
    model_json["scores"] = pass_results
    model_json["results"] = evaluation_result

    output_json[model_name] = model_json

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


def compute_main(dataset_path: str, output_path: str, output_results: bool = True):
    results = []
    with open(output_path, "r") as output_f:
        for line in output_f:
            results.append(json.loads(line))

    with open(dataset_path, "r") as dataset_f:
        dataset = json.load(dataset_f)
    model_name = get_model_name(output_path)
    compute_score(model_name, dataset, results, output_path, output_results)


def main():
    from fire import Fire

    Fire(compute_main)


if __name__ == "__main__":
    main()
