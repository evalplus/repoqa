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
        self,
        question,
        n=1,
        max_tokens=1024,
        temperature=0,
        system_msg=None,
        stop=None,
    ) -> List[str]:
        return self.generate_reply_batched(
            [question],
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            system_msg=system_msg,
            stop=stop,
        )[0]

    def generate_reply_batched(
        self,
        questions: List[str],
        n=1,
        max_tokens=1024,
        temperature=0,
        system_msg=None,
        stop=None,
    ) -> List[List[str]]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"
        if self.tokenizer.chat_template is not None:
            prompts = [
                self.tokenizer.apply_chat_template(
                    construct_message_list(question, system_msg), tokenize=False
                )
                for question in questions
            ]
        else:
            prompts = questions
        # print(prompt)
        vllm_outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            ),
            use_tqdm=True if len(prompts) > 1 else False,
        )

        gen_strs = [
            [output.text.replace("\t", "    ") for output in x.outputs]
            for x in vllm_outputs
        ]
        return gen_strs
