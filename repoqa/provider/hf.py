# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from stop_sequencer import StopSequencer
from transformers import AutoModelForCausalLM, AutoTokenizer

from repoqa.provider.base import BaseProvider
from repoqa.provider.request import construct_message_list, hacky_assistant_stop_seq


class HfProvider(BaseProvider):
    def __init__(self, model, trust_remote_code=False, attn_implementation=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype="auto",
        ).cuda()
        self.stop_sequencer = StopSequencer(
            self.hf_model,
            model_type="causal",  # or seq2seq
            tokenizer=self.tokenizer,
        )
        self.stop_seq = []
        if self.tokenizer.chat_template:
            self.stop_seq.append(hacky_assistant_stop_seq(self.tokenizer))

    @torch.inference_mode()
    def generate_reply(
        self, question, n=1, max_tokens=1024, temperature=0.0, system_msg=None
    ) -> List[str]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"

        prompt_tokens = self.tokenizer.apply_chat_template(
            construct_message_list(question, system_msg),
            return_tensors="pt",
            add_generation_prompt=True,
        ).cuda()

        hf_model = self.hf_model
        input_length = prompt_tokens.size(-1)
        if self.stop_seq:
            hf_model = self.stop_sequencer.register_stop_texts(
                stop_texts=self.stop_seq,
                input_length=input_length,
            )

        gen_args = {"do_sample": False}
        if temperature > 0:
            gen_args["do_sample"] = True
            gen_args["temperature"] = temperature

        output_text = hf_model.generate(
            input_ids=prompt_tokens,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **gen_args,
        )

        gen_strs = [
            self.tokenizer.decode(
                x[input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for x in output_text
        ]
        return gen_strs
