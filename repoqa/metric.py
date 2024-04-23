# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import re

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def compute_function_similarity(
    candidate_function: str, reference_function: str
) -> float:
    candidate_tokens = [item for item in re.split("\s+", candidate_function.strip())]

    reference_tokens = [item for item in re.split("\s+", reference_function.strip())]

    chencherry = SmoothingFunction()

    return sentence_bleu(
        [reference_tokens], candidate_tokens, smoothing_function=chencherry.method4
    )
