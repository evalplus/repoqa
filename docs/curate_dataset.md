# RepoQA Dataset Curation

## Search Needle Functions

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
# collect functions (in-place)
python scripts/curate/function_analysis.py --dataset-path repoqa-{datetime}.json
# select needles (in-place)
python scripts/curate/needle_selection.py --dataset-path repoqa-{datetime}.json
```

> [!Tip]
>
> **Output**: `--dataset-path` (in-place) by adding a `"functions"` field (path to a list function information) for each repo.


### Step 6: Annotate each function with description to make a final dataset

```shell
python scripts/curate/needle_annotation.py --dataset-path repoqa-{datetime}.json
```

> [!Tip]
>
> You need to set `OPENAI_API_KEY` in the environment variable to run GPT-4. But you can enable `--use-batch-api` to save some costs.
>
> **Output**: `--output-desc-path` is a seperate json file specifying the function annotations with its sources.


### Step 7: Merge needle description to the final dataset

```shell
python scripts/curate/merge_annotation.py --dataset-path repoqa-{datetime}.json --annotation-path {output-desc-path}.jsonl
```

> [!Tip]
>
> **Output**: `--dataset-path` (in-place) by adding a `"description"` field for each needle function.
