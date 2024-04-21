# RepoQA Development Notes

## DEV Structure

- `repo`: entrypoint for working repositories
- `repoqa`: source code for the RepoQA evaluation library
- `scripts`: scripts for maintaining the repository and other utilities
  - `dev`: scripts for CI/CD and repository maintenance
  - `curate`: code for dataset curation
    - `dep_analysis`: dependency analysis for different programming languages
  - `cherrypick`: cherry-picked repositories for evaluation
  - `demos`: demos to quickly use some utility functions such as requesting LLMs

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
