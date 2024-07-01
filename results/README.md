## Install GitHub CLI

Check [GitHub CLI](https://github.com/cli/cli) for installation.

## Run evaluations

```bash
export PYTHONPATH=$(pwd)
```

### vLLM

Most OSS models can be evaluated with vLLM.
The tricky part is to tune the `--tensor-parallel-size` (TP) according to the model size.

* `--tensor-parallel-size` can be `1`, `2`, `4`, `8`.
* If your `--tensor-parallel-size` is `2` or `4`, run `nvidia-smi topo -m` to check the GPU connectivity.
* `NV4` means the GPUs are connected with NVLink (which is fast).
* Set `CUDA_VISIBLE_DEVICES` to GPUs with good connectivity.
* Models with < 10B may work with TP-1; 10-20B with TP-2; 20-40B with TP-4; > 40B with TP-8.

```bash
python repoqa/search_needle_function.py --model "mistralai/Mistral-7B-Instruct-v0.2"  --tensor-parallel-size 1 --backend vllm
```

### OpenAI server

```bash
export OPENAI_API_KEY="sk-..." # OAI or DeepSeekCoder key
python repoqa/search_needle_function.py --model "gpt-4o-2024-05-13" --backend openai
python repoqa/search_needle_function.py --model "deepseek-coder" --backend openai
```

### Google Gemini

```bash
export GOOGLE_API_KEY="..."
python repoqa/search_needle_function.py --model "gemini-1.5-pro-latest" --backend google
```

## Pull evaluated results

```bash
cd results
gh release download dev-results --pattern "*.zip" --clobber
# unzip all zip files
unzip "*.zip"
```

## Update model outputs

```bash
cd results
# pull results first
for item in "$(pwd)"/*; do
    # Check if the item is a directory
    if [ -d "$item" ]; then
        # Get the base name of the directory
        dir_name=$(basename "$item")
        zip -FSR "${dir_name}-output.zip" "$dir_name" "*.jsonl"
        zip -FSR "${dir_name}-scores.zip" "$dir_name" "*-SCORES.json"
    fi
done
gh release upload dev-results ./*.zip --clobber
```
