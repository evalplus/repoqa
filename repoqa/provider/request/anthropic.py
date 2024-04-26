# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import signal
import time

import anthropic
from anthropic.types import Message

from repoqa.provider.request import construct_message_list


def make_request(
    client: anthropic.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    system_msg="You are a helpful assistant good at coding.",
    **kwargs,
) -> Message:
    return client.messages.create(
        model=model,
        messages=construct_message_list(message, system_message=system_msg),
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def make_auto_request(client: anthropic.Client, *args, **kwargs) -> Message:
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = make_request(client, *args, **kwargs)
            signal.alarm(0)
        except anthropic.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            signal.alarm(0)
            time.sleep(10)
        except anthropic.APIConnectionError:
            print("API connection error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except anthropic.InternalServerError:
            print("Internal server error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except anthropic.APIError as e:
            print("Unknown API error")
            print(e)
            if (
                e.body["error"]["message"]
                == "Output blocked by content filtering policy"
            ):
                raise Exception("Content filtering policy blocked output")
            signal.alarm(0)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret
