# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json

from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

from repoqa.utility import COMMENT_QUERY, FUNCTION_QUERY, topological_sort

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


# Annotate an incomplete repoqa dataset with function and class information
def main(dataset_path: str, overwrite_analysis: bool = False):
    assert dataset_path.endswith(".json"), "Dataset must be a JSON file, check README"
    with open(dataset_path, "r") as f:
        lists = json.load(f)

    for lang, repos in lists.items():
        assert (
            lang in FUNCTION_QUERY
        ), f"Unsupported language: {lang} -- supported: {FUNCTION_QUERY.keys()}"

        fn_query_text = FUNCTION_QUERY[lang]
        print(f"🔥 Querying {lang} functions with `{fn_query_text}`...")

        parser = get_parser(lang)
        fn_query = get_language(lang).query(fn_query_text)
        fn_name_parser = _cpp_name_parser if lang == "cpp" else _default_name_parser

        for repo in tqdm(repos):
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
                    function_content = code_bytes[node.start_byte : node.end_byte]
                    code_ratio = comment_analysis(function_content, lang)
                    extracted_functions.append(
                        {
                            "name": fn_name_parser(node),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0] + 1,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "global_start_line": global_line_idx + node.start_point[0],
                            "global_end_line": global_line_idx + node.end_point[0] + 1,
                            "global_start_byte": global_byte_idx + node.start_byte,
                            "global_end_byte": global_byte_idx + node.end_byte,
                            "code_ratio": code_ratio,
                        }
                    )
                functions[path] = extracted_functions
                global_byte_idx += len(code)
                global_line_idx += code.count("\n") + 1
            repo["functions"] = functions
            print(
                f"🎉 Found {sum(len(v) for v in functions.values())} functions in {repo['repo']} ({lang})"
            )

    # update the dataset
    with open(dataset_path, "w") as f_out:
        json.dump(lists, f_out)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
