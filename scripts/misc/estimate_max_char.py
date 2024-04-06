# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0


def main(model="deepseek-ai/deepseek-coder-6.7b-instruct", max_tokens=8 * 2**10):
    import time

    import openai

    client = openai.OpenAI(  # Note: if you need UIUC VPN or UIUC network to access the server!
        api_key="none", base_url="http://ise-dynamo.cs.illinois.edu:8888/v1"
    )

    prompt = "def "

    tstart = time.time()
    responses = client.completions.create(
        model=model, prompt=prompt, n=1, max_tokens=max_tokens, temperature=0
    )
    print(f"Time taken: {time.time() - tstart:.1f}s")
    print("Finish reason:", responses.choices[0].finish_reason)
    print(
        "Estimated max context length in #chars is: ",
        len(prompt) + len(responses.choices[0].text),
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
