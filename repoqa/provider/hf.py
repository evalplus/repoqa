# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repoqa.provider.base import BaseProvider
from repoqa.provider.request import construct_message_list


class HfProvider(BaseProvider):
    def __init__(self, model, trust_remote_code=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=trust_remote_code
        ).cuda()

    @torch.inference_mode()
    def generate_reply(
        self, question, n=1, max_tokens=1024, temperature=0, system_msg=None
    ) -> List[str]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"
        prompt_tokens = self.tokenizer.apply_chat_template(
            construct_message_list(question, system_msg), return_tensors="pt"
        ).cuda()
        output_text = self.hf_model.generate(
            input_ids=prompt_tokens,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        gen_strs = [
            self.tokenizer.decode(
                x, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for x in output_text
        ]
        return gen_strs
