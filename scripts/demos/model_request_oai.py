# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

if __name__ == "__main__":
    import openai

    client = openai.OpenAI(  # Note: if you need UIUC VPN or UIUC network to access the server!
        api_key="none", base_url="http://ise-dynamo.cs.illinois.edu:8888/v1"
    )

    task_prefix = "def fibonacci(n):\n"
    prompt = f"""This is the fastest implementation for Fibonacci:
```python
{task_prefix}"""

    # completion
    responses = client.completions.create(
        model="deepseek-ai/deepseek-coder-6.7b-instruct",
        prompt=prompt,
        max_tokens=256,
        n=3,
        stop=["\n```"],
    )

    for c in responses.choices:
        print(task_prefix + c.text)
        print("=" * 8)
