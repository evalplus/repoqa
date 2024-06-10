# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import signal
import time
from typing import List

import openai
from openai.types.chat import ChatCompletion

from repoqa.provider.request import construct_message_list


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    system_msg="You are a helpful assistant good at coding.",
    **kwargs,
) -> ChatCompletion:
    return client.chat.completions.create(
        model=model,
        messages=construct_message_list(message, system_message=system_msg),
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        **kwargs,
    )

def make_embeddings_request(
    client: openai.Client,
    texts: List[str],
    model: str,
) -> List[List[float]]:
    response = client.embeddings.create(input=texts, model=model, encoding_format="float")
    return [d.embedding for d in response.data]

def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")

def make_request_with_retry(func, *args, **kwargs) -> ChatCompletion | List[List[float]]:
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = func(*args, **kwargs)
            signal.alarm(0)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            signal.alarm(0)
            time.sleep(10)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except openai.APIError as e:
            print(e)
            signal.alarm(0)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret

def make_auto_request(*args, **kwargs) -> ChatCompletion:
    return make_request_with_retry(make_request, *args, **kwargs)

def make_auto_embeddings_request(*args, **kwargs) -> List[List[float]]:
    return make_request_with_retry(make_embeddings_request, *args, **kwargs)