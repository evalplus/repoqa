# RepoQA

## The Search-Needle-Function Task

### Inference with Various Backends

#### vLLM

```bash
repoqa.search_needle_function --model "Qwen/CodeQwen1.5-7B-Chat" --caching --backend vllm
```

#### OpenAI Compatible Servers

```bash
repoqa.search_needle_function --base-url "http://api.openai.com/v1" \
                              --model "gpt4-turbo" --caching --backend openai
```

## Read More

* [RepoQA Homepage](https://evalplus.github.io/repoqa.html)
* [RepoQA Dataset Curation](docs/curate_dataset.md)
* [RepoQA Development Notes](docs/dev_note.md)
