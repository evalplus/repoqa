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


> [!Tip]
>
> **Output**: Extend `scripts/cherrypick/lists.json` for a programming language.


### Step 2: Extract repo content

```shell
python scripts/curate/dataset_ensemble_clone.py
```

> [!Tip]
>
> **Output**: `repoqa-{datetime}.json` by adding a `"content"` field (path to content) for each repo.


### Step 3: Dependency analysis

Check [scripts/curate/dep_analysis](scripts/curate/dep_analysis) for more information.

```shell
python scripts/curate/dep_analysis/{language}.py  # python
```

> [!Tip]
>
> **Output**: `{language}.json` (e.g., `python.json`) with a list of items of `{"repo": ..., "commit_sha": ..., "dependency": ...}` field where the dependency is a map of path to imported paths.

> [!Note]
>
> The `{language}.json` should be uploaded as a release.
>
> To fetch the release, go to `scripts/curate/dep_analysis/data` and run `gh release download dependency --pattern "*.json" --clobber`.


### Step 4: Merge step 2 and step 3

```shell
python scripts/curate/merge_dep.py --dataset-path repoqa-{datetime}.json
```

> [!Tip]
>
> **Input**: Download dependency files in to `scripts/curate/dep_analysis/data`.
>
> **Output**: Update `repoqa-{datetime}.json` by adding a `"dependency"` field for each repository.


### Step 5: Function collection with TreeSitter

```shell
# collect functions
python scripts/curate/function_analysis.py --dataset-path repoqa-{datetime}.json
# select needles
python scripts/curate/function_selection.py --dataset-path repoqa-{datetime}.json
```

> [!Tip]
>
> **Output**: `--dataset-path` (inplace) by adding a `"functions"` field (path to a list function information) for each repo.


### Step 6: QA annootation

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
