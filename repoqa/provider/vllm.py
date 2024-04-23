# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from repoqa.provider.base import BaseProvider
from repoqa.provider.request import construct_message_list


class VllmProvider(BaseProvider):
    def __init__(
        self, model, tensor_parallel_size, max_model_len, trust_remote_code=False
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )

    def generate_reply(
        self, question, n=1, max_tokens=1024, temperature=0, system_msg=None
    ) -> List[str]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"
        prompt = self.tokenizer.apply_chat_template(
            construct_message_list(question, system_msg), tokenize=False
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
