# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def compute_function_similarity(
    candidate_function: str, reference_function: str
) -> float:
    candidate_tokens = [
        item
        for line in candidate_function.strip().split("\n")
        if line.strip()
        for item in line.strip().split()
    ]

    reference_tokens = [
        item
        for line in reference_function.strip().split("\n")
        if line.strip()
        for item in line.strip().split()
    ]

    chencherry = SmoothingFunction()

    return sentence_bleu(
        [reference_tokens], candidate_tokens, smoothing_function=chencherry.method7
    )
