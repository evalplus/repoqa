#!/bin/bash

# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

set -e

export PYTHONPATH=$(pwd)

for path in "$(pwd)"/results/**/*.jsonl; do
    # if the file size is greater than 10MB
    file_size_mb=$(du -m "$path" | cut -f1)
    echo "Size of $path: $file_size_mb MB"
    if [ $file_size_mb -lt 10 ]; then
        echo "File size is less than 10MB. Skipping..."
        continue
    fi
    yes | python repoqa/compute_score.py --model-output-path $path
done
