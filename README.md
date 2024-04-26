# RepoQA: Evaluating Long-Context Code Understanding

🏠 Homepage: [https://evalplus.github.io/repoqa.html](https://evalplus.github.io/repoqa.html)

## 🚀 Installation

```bash
# without vLLM (can run openai, anthropic, and huggingface backends)
pip install --upgrade repoqa
# To enable vLLM
pip install --upgrade "repoqa[vllm]"
```

<details><summary>⏬ Install nightly version <i>:: click to expand ::</i></summary>
<div>

```bash
pip install --upgrade "git+https://github.com/evalplus/repoqa.git"                 # without vLLM
pip install --upgrade "repoqa[vllm] @ git+https://github.com/evalplus/repoqa@main" # with vLLM
```

</div>
</details>

<details><summary>⏬ Using RepoQA as a local repo? <i>:: click to expand ::</i></summary>
<div>

```bash
git clone https://github.com/evalplus/repoqa.git
cd repoqa
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -r requirements.txt
```

</div>
</details>

## 🏁 Search Needle Function (SNF)

You can run the SNF evaluation using various backends.

> [!Note]
>
> All evaluation can be performed in one just command.
>
> As a reference of evaluation time, it takes 30 minutes to evaluate a 7B model using two A6000s.

### OpenAI Compatible Servers

```bash
repoqa.search_needle_function --model "gpt4-turbo" --caching --backend openai
# 💡 If you use customized server such vLLM:
# repoqa.search_needle_function --base-url "http://url.to.vllm.server/v1" \
#                               --model "gpt4-turbo" --caching --backend openai
```

### Anthropic Compatible Servers

```bash
repoqa.search_needle_function --model "claude-3-haiku-20240307" --caching --backend anthropic
```

### vLLM

```bash
repoqa.search_needle_function --model "Qwen/CodeQwen1.5-7B-Chat" \
                              --caching --backend vllm
```

### HuggingFace transformers

```bash
repoqa.search_needle_function --model "Qwen/CodeQwen1.5-7B-Chat" \
                              --caching --backend hf --trust-remote-code
```

### Google Generative AI API (Gemini)

```bash
repoqa.search_needle_function --model "gemini-1.5-pro-latest" --caching --backend google
```

> [!Tip]
>
> To evaluate models whose context size is smaller than the prompt, you can edit the `config.json` file to modify `max_position_embeddings` for the model in HuggingFace cache directory.

### Usage

> [!Tip]
>
> - **Input**:
>   - `--model`: Hugging-Face model ID, such as `ise-uiuc/Magicoder-S-DS-6.7B`
>   - `--backend`: `vllm` (default) or `openai`
>   - `--base-url`: OpenAI API base URL
>   - `--code-context-size` (default: 16384): #tokens (by DeepSeekCoder tokenizer) of repository context
>   - `--caching` (default: True): accelerate subsequent runs by caching tokenization and chuncking results
>   - `--max-new-tokens` (default: 1024): Maximum #new tokens to generate
>   - `--system-message` (default: None): system message (note it's not supported by some models)
>   - `--tensor-parallel-size`: #GPUS for doing tensor parallelism (only for vLLM)
>   - `--languages` (default: None): List of languages to evaluate (None means all)
>   - `--result-dir` (default: "results"): Directory to save the model outputs and evaluation results
>   - `--trust-remote-code` (default: False): allow remote code (for HuggingFace transformers and vLLM)
> - **Output**:
>   - `results/ntoken_{code-context-size}/{model}.jsonl`: Model generated outputs
>   - `results/ntoken_{code-context-size}/{model}-SCORE.json`: Evaluation results

### Compute Scores

By default, the `repoqa.search_needle_function` command will evaluate model outputs and compute scores after text generation.
However, you can also separately compute scores using the following command:

```shell
repoqa.compute_score --model-output-path={model-output}.jsonl
```

> [!Tip]
>
> - **Input**: Path to the model generated outputs.
> - **Output**: The evaluation scores would be stored in `{model-output}-SCORES.json`

## 📚 Read More

- [RepoQA Homepage](https://evalplus.github.io/repoqa.html)
- [RepoQA Dataset Curation](docs/curate_dataset.md)
- [RepoQA Development Notes](docs/dev_note.md)
