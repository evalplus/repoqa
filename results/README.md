## Install GitHub CLI

Check [GitHub CLI](https://github.com/cli/cli) for installation.

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
