# RepoQA: Evaluating Long-Context Code Understanding

<p align="center">
    <a href="#-installation">🚀 Installation</a> •
    <a href="#-search-needle-function">🏁 Search Needle Function</a> •
    <a href="#-compute-scores"> 🧮 Compute Score</a> •
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

### Inference

#### vLLM

```bash
repoqa.search_needle_function --model "Qwen/CodeQwen1.5-7B-Chat" --caching --backend vllm
```

#### OpenAI Compatible Servers

```bash
repoqa.search_needle_function --base-url "http://api.openai.com/v1" \
                              --model "gpt4-turbo" --caching --backend openai
```

<details><summary>⌨️ Command-line arguments <i>:: click to expand ::</i></summary>
<div>

* `--model`: Hugging-Face model ID, such as `ise-uiuc/Magicoder-S-DS-6.7B`
* `--backend`: `vllm` (default) or `openai`
* `--base-url`: OpenAI API base URL
* `--code-context-size` (default: 16384): Number of tokens (using DeepSeekCoder tokenizer) of code in the long context
* `--caching` (default: False): if enabled, the tokenization and chuncking results will be cached to accelerate subsequent runs
* `--max-new-tokens` (default: 1024): Maximum number of new tokens to generate
* `--system-message` (default: None): if given, the model use a system message (but note some models don't support system message)
* `--tensor-parallel-size`: Number of tensor parallelism (only for vLLM)
* `--languages` (default: None): List of languages to evaluate (None means all)
* `--result-dir` (default: "results"): Directory to save the model outputs and evaluation results
* `--store-score` (default: False): if enabled, computed score will be stored within result-dir with name `{model-info}-SCORE.jsonl`

</div>
</details>

## 🧮 Compute Scores

```shell
python repoqa/compute_score.py --dataset_path repoqa-{datetime}.json --output_path={model-evaluation-output}.jsonl --output_results True
```

<details><summary>⌨️ Output Information</i></summary>
<div>

- `Output`: The output of score evaluation results would be stored in `{model-evaluation-output}-SCORES.jsonl`

- `Stdout`: The pass@1 results for all languages and each language at each similarity threshold would also be printed out.

</div>
</details>


## 📚 Read More

* [RepoQA Homepage](https://evalplus.github.io/repoqa.html)
* [RepoQA Dataset Curation](docs/curate_dataset.md)
* [RepoQA Development Notes](docs/dev_note.md)
