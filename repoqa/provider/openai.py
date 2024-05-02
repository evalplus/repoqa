# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

from openai import Client
from transformers import AutoTokenizer

from repoqa.provider.base import BaseProvider
from repoqa.provider.request import hacky_assistant_stop_seq
from repoqa.provider.request.openai import make_auto_request


class OpenAIProvider(BaseProvider):
    def __init__(self, model, base_url: str = None):
        self.model = model
        self.client = Client(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=base_url
        )
        self.stop_seq = []
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            if tokenizer.chat_template:
                self.stop_seq.append(hacky_assistant_stop_seq(tokenizer))
            print("Using stop sequence: ", self.stop_seq)
        except:
            print("Failed to automatically fetch stop tokens from HuggingFace.")

    def generate_reply(
        self, question, n=1, max_tokens=1024, temperature=0.0, system_msg=None
    ) -> List[str]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"
        replies = make_auto_request(
            self.client,
            message=question,
            model=self.model,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            system_msg=system_msg,
            stop=self.stop_seq,
        )

        return [reply.message.content for reply in replies.choices]
