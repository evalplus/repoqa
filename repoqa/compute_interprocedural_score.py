# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from repoqa.utility import get_model_name, progress, save_json

CAPTURE_HEAD = "<path_start>"
CAPTURE_TAIL = "<path_end>"

# TODO: add pretty display table to show results


def path_evaluator(groundtruth: List[str], output: str) -> Tuple[bool, int]:
    output_result = output.split(CAPTURE_HEAD)[-1].split(CAPTURE_TAIL)[0]
    output_path = output_result.split(",")
    groundtruth_set = set(groundtruth)
    overlap = 0
    correct_path = True
    for index, function in enumerate(output_path):
        stripped_function = function.strip()
        if index >= len(groundtruth):
            correct_path = False
            break
        if groundtruth[index] != stripped_function:
            correct_path = False
        if stripped_function in groundtruth_set:
            overlap += 1
            groundtruth_set.remove(stripped_function)

    return (correct_path, float(overlap / len(groundtruth)))


def compute_interprocedural_score(model_name, model_outputs) -> Dict:
    evaluation_result = defaultdict(list)
    with progress(f"Scoring {model_name}") as pbar:
        for result in pbar.track(model_outputs):
            lang = result["language"]
            repo_name = result["repo"]
            outputs = result["output"]
            ground_truth = result["groundtruth_path"]

            output = outputs[0]
            verdict, path_overlap = path_evaluator(ground_truth, output)

            current_task = {
                "repo": repo_name,
                "start": result["start"],
                "end": result["end"],
                "ground_truth": ground_truth,
                "output": output,
                "correct_path": verdict,
                "path_overlap": path_overlap,
            }
            evaluation_result[lang].append(current_task)

    for lang, results in evaluation_result.items():
        correct_accuracy = 0
        overlap_score = 0
        for result in results:
            correct_accuracy += 1 if result["correct_path"] else 0
            overlap_score += result["path_overlap"]
        print(
            f"Score for {lang} for {model_name}. Total Accuracy = {float(correct_accuracy/len(results))}. Overlap Score = {overlap_score/len(results)}"
        )
    return evaluation_result


def compute_main(model_output_path: str):
    model_outputs = []
    with open(model_output_path, "r") as output_f:
        for line in output_f:
            model_outputs.append(json.loads(line))

    file_base, _ = os.path.splitext(model_output_path)
    result_path = file_base + "-SCORES.json"
    model_name = get_model_name(model_output_path)
    output_json = compute_interprocedural_score(model_name, model_outputs)
    save_json(output_json, result_path)


def main():
    from fire import Fire

    Fire(compute_main)


if __name__ == "__main__":
    main()
