# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

# https://github.com/evalplus/repoqa/issues/25

set -x

python repoqa/search_needle_function.py --dataset-path repoqa-2024-04-16.json --tensor-parallel-size 2 --model "meta-llama/Meta-Llama-3-8B-Instruct"
python repoqa/search_needle_function.py --dataset-path repoqa-2024-04-16.json --tensor-parallel-size 2 --model "deepseek-ai/deepseek-coder-6.7b-instruct"
python repoqa/search_needle_function.py --dataset-path repoqa-2024-04-16.json --tensor-parallel-size 2 --model "codellama/CodeLlama-7b-Instruct-hf"
python repoqa/search_needle_function.py --dataset-path repoqa-2024-04-16.json --tensor-parallel-size 2 --model "mistralai/Mistral-7B-Instruct-v0.2"
