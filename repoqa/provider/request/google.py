# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import signal
import time

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

from repoqa.provider.request import construct_message_list


def make_request(
    client: genai.GenerativeModel,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    system_msg="You are a helpful assistant good at coding.",
    **kwargs,
) -> genai.types.GenerateContentResponse:
    messages = []
    if system_msg:
        messages.append({"role": "system", "parts": [system_msg]})
    messages.append({"role": "user", "parts": [message]})
    return client.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            candidate_count=n, max_output_tokens=max_tokens, temperature=temperature
        ),
        **kwargs,
    )


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def make_auto_request(*args, **kwargs) -> genai.types.GenerateContentResponse:
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = make_request(*args, **kwargs)
            signal.alarm(0)
        except ResourceExhausted as e:
            print("Rate limit exceeded. Waiting...", e.message)
            signal.alarm(0)
            time.sleep(10)
        except GoogleAPICallError as e:
            print(e.message)
            signal.alarm(0)
            time.sleep(1)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret
