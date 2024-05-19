# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

# Run with ```python scripts/curate/obfuscate_nl.py repoqa-2024-04-20.json```
# Will save to repoqa-2024-04-20-obfuscated.json

import json
import os
import re

from fire import Fire
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

from repoqa.utility import COMMENT_QUERY, FUNCTION_NAME_QUERY


def remove_comments(code, language):
    query_texts = COMMENT_QUERY[language]
    parser = get_parser(language)
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)
    comment_chunks = []
    for query_text in query_texts:
        comment_query = get_language(language).query(query_text)
        for node, _ in comment_query.captures(tree.root_node):
            comment_chunks.append(node.text.decode("utf-8"))
    comment_chunks.sort(key=len, reverse=True)
    for chunk in comment_chunks:
        chunk_lines = chunk.splitlines()
        chunk_lines_len = [len(bytes(line, "utf-8")) for line in chunk_lines]
        chunk_lines_empty = [
            (bytes("", "utf-8").ljust(llen, b"\0")).decode("utf-8")
            for llen in chunk_lines_len
        ]
        chunk_empty = "\0".join(chunk_lines_empty)
        chunk_empty = chunk_empty[:-1] + "\n"
        code = code.replace(chunk, chunk_empty)
    return code


def rename_functions(code, language, starting_index=0):
    func_name_query = get_language(language).query(FUNCTION_NAME_QUERY[language])
    parser = get_parser(language)
    print(f"Running rename_functions: {code}, {language}")
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)
    function_names = set()
    for capture in func_name_query.captures(tree.root_node):
        node, _ = capture
        function_names.add(node.text.decode("utf-8"))
    function_map = {}
    current_index = starting_index
    for name in function_names:
        function_map[name] = f"function_{starting_index}"
        code = code.replace(name, function_map[name])
        current_index += 1
    return code, function_map


def main(ds_filepath: str):
    dataset_file = open(ds_filepath, "r")
    dataset = dataset_file.read()
    dataset = json.loads(dataset)
    dataset_file.close()

    for lang in dataset.keys():
        print(f"🔥 Processing language: {lang}")
        for repo_idx in tqdm(range(len(dataset[lang]))):
            for filepath in dataset[lang][repo_idx]["content"].keys():
                prev_byte_len = len(
                    bytes(dataset[lang][repo_idx]["content"][filepath], "utf-8")
                )
                dataset[lang][repo_idx]["content"][filepath] = remove_comments(
                    dataset[lang][repo_idx]["content"][filepath], lang
                )
                new_byte_len = len(
                    bytes(dataset[lang][repo_idx]["content"][filepath], "utf-8")
                )
                assert prev_byte_len == new_byte_len

    dataset_dir = "/".join(ds_filepath.split("/")[:-1])
    ds_filepath = ds_filepath.split("/")[-1]
    ds_fname = ".".join(ds_filepath.split(".")[:-1])
    ds_ext = ds_filepath.split(".")[-1]

    obfs_ds_file = open(
        os.path.join(dataset_dir, f"{ds_fname}-obfuscated.{ds_ext}"), "w+"
    )
    obfs_ds_file.write(json.dumps(dataset))
    obfs_ds_file.close()


if __name__ == "__main__":
    Fire(main)
