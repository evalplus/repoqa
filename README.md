# RepoQA

## DEV Structure

- `repo`: entrypoint for working repositories
- `repoqa`: source code for the RepoQA evaluation library
- `scripts`: scripts for maintaining the repository and other utilities
  - `dev`: scripts for CI/CD and repository maintenance
  - `curate`: code for dataset curation
    - `dep_analysis`: dependency analysis for different programming languages
  - `cherrypick`: cherry-picked repositories for evaluation
  - `demos`: demos to quickly use some utility functions such as requesting LLMs

## Making a dataset

### Step 1: Cherry-pick repositories

See [scripts/cherrypick/README.md](cherrypick/README.md) for more information.


> [!Note]
>
> **Output**: Extend `scripts/cherrypick/lists.json` for a programming language.

### Step 2: Extract repo content

```shell
python scripts/curate/dataset_ensemble_clone.py
```

> [!Note]
>
> **Output**: `repoqa-{datetime}.json` by adding a `"content"` field (path to content) for each repo.

### Step 3: Dependency analysis

Check [scripts/curate/dep_analysis](scripts/curate/dep_analysis) for more information.

```shell
# python
python scripts/curate/dep_analysis/python.py --dataset-path repoqa-{datetime}.json
```

> [!Note]
>
> **Output**: `--dataset-path` (inplace) by adding a `"dependency"` field (path to a list imported path) for each repo.


### Step 4: Needle function collection

TBD.

### Step 5: QA annootation

TBD.

## Development Beginner Notice

### After clone

```shell
pip install pre-commit
pre-commit install
pip install -r requirements.txt
pip install -r scripts/curate/requirements.txt
```

### Import errors?

```shell
# Go to the root path of RepoQA
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
