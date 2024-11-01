# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict

from tqdm import tqdm
from tree_sitter import Node
from tree_sitter_languages import get_language, get_parser

from repoqa.utility import (
    CLASS_TYPE_NODE,
    COMMENT_QUERY,
    FUNCTION_QUERY,
    topological_sort,
)

_default_name_parser = lambda node: node.child_by_field_name("name").text.decode()
_cpp_name_parser = (
    lambda node: node.child_by_field_name("declarator")
    .child_by_field_name("declarator")
    .text.decode()
)


def comment_analysis(code: bytes, language: str) -> float:
    query_texts = COMMENT_QUERY[language]
    parser = get_parser(language)
    tree = parser.parse(code)
    characters = 0
    for query_text in query_texts:
        comment_query = get_language(language).query(query_text)
        for node, _ in comment_query.captures(tree.root_node):
            comment_text = code[node.start_byte : node.end_byte]
            characters += len(comment_text)
    return characters / len(code)


# For cpp, java, typescript and python we find the class of the method, if one exists
# for go and rust, we find the type
def class_type_analysis(current_node: Node, lang: str) -> str:
    while current_node:
        if current_node.type in CLASS_TYPE_NODE[lang]:
            class_name_node = current_node.child_by_field_name("name")
            if class_name_node:
                return class_name_node.text.decode("utf-8")
            return "not found"
        current_node = current_node.parent
    return ""


# Annotate an incomplete repoqa dataset with function and class information
def main(dataset_path: str, overwrite_analysis: bool = False):
    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    with open(dataset_path, "r") as f:
        lists = json.load(f)

    for lang, repos in lists.items():
        # TODO: remove
        if lang != "python":
            continue
        assert (
            lang in FUNCTION_QUERY
        ), f"Unsupported language: {lang} -- supported: {FUNCTION_QUERY.keys()}"

        fn_query_text = FUNCTION_QUERY[lang]
        print(f"🔥 Querying {lang} functions with `{fn_query_text}`...")

        parser = get_parser(lang)
        fn_query = get_language(lang).query(fn_query_text)
        fn_name_parser = _cpp_name_parser if lang == "cpp" else _default_name_parser

        for repo in tqdm(repos):
            function_counts = {}
            # skip if the repo already has function information
            if not overwrite_analysis and repo.get("functions"):
                continue

            if not repo.get("dependency"):
                print(
                    f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `dependency` -- do dependency analysis first"
                )
                continue

            ordered_paths = topological_sort(repo["dependency"])
            global_byte_idx = 0
            global_line_idx = 0
            functions = {}  # path to a list of functions
            for path in ordered_paths:
                code = repo["content"][path]
                code_bytes = bytes(code, "utf8")
                tree = parser.parse(code_bytes)
                extracted_functions = []
                for capture in fn_query.captures(tree.root_node):
                    node, _ = capture
                    # TODO: fix this for go
                    if lang == "go":
                        function_class_type = ""
                    else:
                        function_class_type = class_type_analysis(node, lang)
                    function_name = fn_name_parser(node)
                    function_content = code_bytes[node.start_byte : node.end_byte]
                    code_ratio = comment_analysis(function_content, lang)
                    if function_class_type:
                        full_name = (
                            path + "::" + function_class_type + "." + function_name
                        )
                    else:
                        full_name = path + "::" + function_name
                    extracted_functions.append(
                        {
                            "name": function_name,
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0] + 1,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "global_start_line": global_line_idx + node.start_point[0],
                            "global_end_line": global_line_idx + node.end_point[0] + 1,
                            "global_start_byte": global_byte_idx + node.start_byte,
                            "global_end_byte": global_byte_idx + node.end_byte,
                            "code_ratio": code_ratio,
                            "file": path,
                            "class_type": function_class_type,
                            "full_name": full_name,
                        }
                    )
                    function_counts[full_name] = (
                        function_counts.get(function_name, 0) + 1
                    )
                functions[path] = extracted_functions
                global_byte_idx += len(code)
                global_line_idx += code.count("\n") + 1

            # Update whether function name is unique
            unique_count = 0
            for _, file_functions in functions.items():
                for function in file_functions:
                    function["is_unique"] = function_counts[function["full_name"]] == 1
                    if function_counts[function["full_name"]] == 1:
                        unique_count += 1

            repo["functions"] = functions
            print(
                f"🎉 Found {sum(len(v) for v in functions.values())} functions in {repo['repo']} ({lang})"
            )
            print(f"🎉 Found {unique_count} unique functions in {repo['repo']} ({lang})")

    # update the dataset
    with open(dataset_path, "w") as f_out:
        json.dump(lists, f_out)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
