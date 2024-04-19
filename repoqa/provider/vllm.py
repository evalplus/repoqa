# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from repoqa.provider.base import SYSTEM_MSG, BaseProvider


class VllmProvider(BaseProvider):
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model=model)

    def generate_reply(
        self, question, n=1, max_tokens=1024, temperature=0
    ) -> List[str]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": question},
            ],
            tokenize=False,
        )
        vllm_outputs = self.llm.generate(
            [prompt],
            SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs
