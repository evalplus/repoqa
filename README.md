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

Search Needle Function is the first and base RepoQA task which aims to practice LLMs' ability of **long-context code understanding and retrieval**.
Its corresponding real-life scenario is to perform precise code search from function description.

<details><summary>🔎 More dataset details <i>:: click to expand ::</i></summary>
<div>

> [!Note]
>
> SNF includes 500 tests (5 programming languages x 10 repos x 10 needle functions) where an LLM is given:
>
> 1. A large code context sorted in file dependency
> 2. A NL description of the needle function without revealing keywords like function names
> 3. An instruction to retrieve the described function
>
> The evaluator passes a test if the searched function is syntactically closest to the ground-truth compared against
> other functions (systematically parsed by `treesitter`) and the similarity is greater than a user defined threshold (by default 0.8).

</div>
</details>

You can run the SNF evaluation using various backends:

### OpenAI Compatible Servers

```bash
repoqa.search_needle_function --model "gpt4-turbo" --backend openai
# 💡 If you use openai API compatible server such as vLLM servers:
# repoqa.search_needle_function --base-url "http://localhost:[PORT]/v1" \
#                               --model "Qwen/CodeQwen1.5-7B-Chat" --backend openai
```

### Anthropic Compatible Servers

```bash
repoqa.search_needle_function --model "claude-3-haiku-20240307" --backend anthropic
```

### vLLM

```bash
repoqa.search_needle_function --model "Qwen/CodeQwen1.5-7B-Chat" --backend vllm
```

<details><summary>🔎 Context extension for small-ctx models <i>:: click to expand ::</i></summary>
<div>

> There are two ways to unlock a model's context at inference time:
>
> 1. **Direct Extension**: Edit `max_positional_embedding` of the model's `config.json` (e.g., `hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/[hash]/config.json`) to something like `22528`.
> 2. **[Dynamic RoPE Scaling](https://blog.eleuther.ai/yarn/#dynamic-scaling)**:
>    To extend `Meta-Llama-3-8B-Instruct` from 8k to 32k (4x), edit the `config.json`:
>
> `"rope_scaling": {"type": "dynamic", "factor": 4.0}`
>
> Note: This works for vLLM `<0.4.3` and HuggingFace transformers. RepoQA will automatically configure dynamic RoPE for vLLM `>= 0.4.3`

</div>
</details>

> [!Note]
>
> Reference evaluation time:
>
> - Llama3-8B-Instruct: 45 minutes on 2xA6000 (PCIe NVLink)
> - Llama3-70B-Instruct: 100 minutes on 4xA100 (PCIe NVLink)

### HuggingFace transformers

```bash
repoqa.search_needle_function --model "Qwen/CodeQwen1.5-7B-Chat" --backend hf --trust-remote-code
```

> [!Tip]
>
> Installing [flash-attn](https://github.com/Dao-AILab/flash-attention) and
> additionally set `--attn-implementation "flash_attention_2"` can largely
> lower the memory requirement.

<details><summary>🔨 Having trouble installing `flash-attn`? <i>:: click to expand ::</i></summary>
<div>

> If you have trouble with `pip install flash-attn --no-build-isolation`,
> you can try to directly use [pre-built wheels](https://github.com/Dao-AILab/flash-attention/releases):
>
> ```shell
> export FLASH_ATTN_VER=2.5.8 # check latest version at https://github.com/Dao-AILab/flash-attention/releases
> export CUDA_VER="cu122"     # check available ones at https://github.com/Dao-AILab/flash-attention/releases
> export TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
> export PY_VER=$(python -c "import platform; print(''.join(platform.python_version().split('.')[:2]))")
> export OS_ARCH=$(python -c "import platform; print(f'{platform.system().lower()}_{platform.machine()}')")
>
> export WHEEL=flash_attn-${FLASH_ATTN_VER}+${CUDA_VER}torch${TORCH_VER}cxx11abiFALSE-cp${PY_VER}-cp${PY_VER}-${OS_ARCH}.whl
> wget https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/${WHEEL}
> pip install ${WHEEL}
> ```

</div>
</details>

### Google Generative AI API (Gemini)

```bash
repoqa.search_needle_function --model "gemini-1.5-pro-latest" --backend google
```

### CLI Usage

- **Input**:
  - `--model`: Hugging-Face model ID, such as `ise-uiuc/Magicoder-S-DS-6.7B`
  - `--backend`: `vllm` (default) or `openai`
  - `--base-url`: OpenAI API base URL
  - `--code-context-size` (default: 16384): #tokens (by DeepSeekCoder tokenizer) of repository context
  - `--caching` (default: True): accelerate subsequent runs by caching preprocessing; `--nocaching` to disable
  - `--max-new-tokens` (default: 1024): Maximum #new tokens to generate
  - `--system-message` (default: None): system message (note it's not supported by some models)
  - `--tensor-parallel-size`: #GPUS for doing tensor parallelism (only for vLLM)
  - `--languages` (default: None): List of languages to evaluate (None means all)
  - `--result-dir` (default: "results"): Directory to save the model outputs and evaluation results
  - `--ignore-comments` (default: False): During evaluation, ignore groundtruth and model comments
  - `--trust-remote-code` (default: False): allow remote code (for HuggingFace transformers and vLLM)
  - `--attn-implementation` (default: None): Use "flash_attention_2" if your HF hits OOM
- **Output**:
  - `results/ntoken_{code-context-size}/{model}.jsonl`: Model generated outputs
  - `results/ntoken_{code-context-size}/{model}-SCORE.json`: Evaluation results

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
