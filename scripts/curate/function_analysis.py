# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json

from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
}

_default_name_parser = lambda node: node.child_by_field_name("name").text.decode()
_cpp_name_parser = (
    lambda node: node.child_by_field_name("declarator")
    .child_by_field_name("declarator")
    .text.decode()
)


def topological_sort(graph):
    # Stack to store the topological order
    stack = []
    # Set to keep track of visited nodes
    visited = set()

    # Recursive function to process nodes
    def dfs(node):
        # Mark the current node as visited
        visited.add(node)
        # Recurse for all the vertices adjacent to this vertex
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                dfs(neighbour)
        # Push current vertex to stack which stores the result
        stack.append(node)

    # Call the recursive helper function to store the topological sort starting from all vertices one by one
    for node in graph:
        if node not in visited:
            dfs(node)

    return stack


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
            functions = {}  # path to a list of functions
            for path in ordered_paths:
                code = repo["content"][path]
                tree = parser.parse(bytes(code, "utf8"))
                extracted_functions = []
                for capture in fn_query.captures(tree.root_node):
                    node, _ = capture
                    extracted_functions.append(
                        {
                            "name": fn_name_parser(node),
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "global_start_byte": global_byte_idx + node.start_byte,
                            "global_end_byte": global_byte_idx + node.end_byte,
                        }
                    )
                functions[path] = extracted_functions
                global_byte_idx += len(code)
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
