# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import os

import tempdir
import wget
from appdirs import user_cache_dir

CACHE_DIR = user_cache_dir("repoqa")

REPOQA_DATA_OVERRIDE_PATH = os.getenv("REPOQA_DATA_OVERRIDE_PATH", None)
REPOQA_DATA_VERSION = os.getenv("REPOQA_DATA_VERSION", "2024-04-20")


def _get_repoqa_data_ready_path() -> str:
    if REPOQA_DATA_OVERRIDE_PATH:
        assert os.path.exists(
            REPOQA_DATA_OVERRIDE_PATH
        ), f"File not found: {REPOQA_DATA_OVERRIDE_PATH}"
        return REPOQA_DATA_OVERRIDE_PATH

    gzip_url = f"https://github.com/evalplus/repoqa_release/releases/download/{REPOQA_DATA_VERSION}/repoqa-{REPOQA_DATA_VERSION}.json.gz"
    cache_path = os.path.join(CACHE_DIR, f"repoqa-{REPOQA_DATA_VERSION}.json")
    # Check if human eval file exists in CACHE_DIR
    if not os.path.exists(cache_path):
        # Install HumanEval dataset and parse as json
        print(f"Downloading dataset from {gzip_url}")
        with tempdir.TempDir() as tmpdir:
            gzip_path = os.path.join(tmpdir, f"data.json.gz")
            wget.download(gzip_url, gzip_path)

            with gzip.open(gzip_path, "rb") as f:
                repoqa_data = f.read().decode("utf-8")

        # create CACHE_DIR if not exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Write the original human eval file to CACHE_DIR
        with open(cache_path, "w") as f:
            f.write(repoqa_data)

    return cache_path


def get_repoqa_data():
    with open(_get_repoqa_data_ready_path(), "r") as f:
        return json.load(f)
