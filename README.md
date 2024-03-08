# RepoQA

## DEV Structure

- `repo`: entrypoint for working repositories
- `repoqa`: source code for the RepoQA evaluation library
- `scripts`: scripts for maintaining the repository and other utilities
  - `dev`: scripts for CI/CD and repository maintenance
  - `curate`: code for dataset curation
  - `cherrypick`: cherry-picked repositories for evaluation

## Development Beginner Notice

### After clone

```shell
pip install pre-commit
pre-commit install
pip install -r requirements.txt
```

### Import errors?

```shell
# Go to the root path of RepoQA
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
