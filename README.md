# RepoQA: Evaluating Long-Context Code Understanding

<p align="center">
    <a href="#-installation">🚀 Installation</a> •
    <a href="#-search-needle-function">🏁 Search Needle Function</a> •
    <a href="#-read-more">📚 Read More</a>
</p>

## 🚀 Installation

```bash
pip install repoqa
```

<details><summary>⏬ Install nightly version <i>:: click to expand ::</i></summary>
<div>

```bash
pip install "git+https://github.com/evalplus/repoqa.git" --upgrade
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


## 🏁 Search Needle Function

### Inference with vLLM

```bash
repoqa.search_needle_function --model "Qwen/CodeQwen1.5-7B-Chat" --caching --backend vllm
```

### Inference with OpenAI Compatible Servers

```bash
repoqa.search_needle_function --base-url "http://api.openai.com/v1" \
                              --model "gpt4-turbo" --caching --backend openai
```

### Inference with HuggingFace transformers

```bash
repoqa.search_needle_function --model "gpt2" "Qwen/CodeQwen1.5-7B-Chat" --caching --backend hf
```

### Usage

> [!Tip]
>
> * **Input**:
>   * `--model`: Hugging-Face model ID, such as `ise-uiuc/Magicoder-S-DS-6.7B`
>   * `--backend`: `vllm` (default) or `openai`
>   * `--base-url`: OpenAI API base URL
>   * `--code-context-size` (default: 16384): Number of tokens (using DeepSeekCoder tokenizer) of code in the long context
>   * `--caching` (default: False): if enabled, the tokenization and chuncking results will be cached to accelerate subsequent runs
>   * `--max-new-tokens` (default: 1024): Maximum number of new tokens to generate
>   * `--system-message` (default: None): if given, the model use a system message (but note some models don't support system message)
>   * `--tensor-parallel-size`: Number of tensor parallelism (only for vLLM)
>   * `--languages` (default: None): List of languages to evaluate (None means all)
>   * `--result-dir` (default: "results"): Directory to save the model outputs and evaluation results
> * **Output**:
>   * `results/ntoken_{code-context-size}/{model}.jsonl`: Model generated outputs
>   * `results/ntoken_{code-context-size}/{model}-SCORE.json`: Evaluation scores (also see [Compute Scores](#compute-scores))

### Compute Scores

By default, the `repoqa.search_needle_function` command will also compute scores after producing model outputs.
However, you can also compute scores separately using the following command:

```shell
repoqa.compute_score --model-output-path={model-output}.jsonl
```

> [!Tip]
>
> * **Input**: Path to the model generated outputs.
> * **Output**: The evaluation scores would be stored in `{model-output}-SCORES.json`


## 📚 Read More

* [RepoQA Homepage](https://evalplus.github.io/repoqa.html)
* [RepoQA Dataset Curation](docs/curate_dataset.md)
* [RepoQA Development Notes](docs/dev_note.md)
