# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "((function_definition(function_declarator) (_))) @fdef",
    "go": "(function_declaration name: (_)) @fdef",
}

COMMENT_QUERY = {
    "python": [
        "(block (expression_statement (string) @docstring))",
        "(comment) @comment",
    ],
    "java": ["(line_comment) @comment", "(block_comment) @comment"],
    "cpp": ["(comment) @comment"],
    "rust": ["(line_comment) @comment", "(block_comment) @comment"],
    "typescript": ["(comment) @comment"],
    "go": ["(comment) @comment"],
}

FUNCTION_NAME_QUERY = {
    "python": """
        ((function_definition
          name: (identifier) @function_name))
    """,
    "java": """
        (method_declaration
          name: (identifier) @method_name)
    """,
    "typescript": """
        (function_declaration
          name: (identifier) @function_name)
    """,
    "rust": """
        (function_item
          name: (identifier) @function_name)
    """,
    "cpp": """
        (function_definition
          name: (identifier) @function_name)
    """,
}

CLASS_TYPE_NODE = {
    "python": ["class_definition"],
    "java": ["class_declaration"],
    "cpp": ["class_specifier", "struct_specifier"],
    "go": [],  # For go, we need a different mechanism to check, sigh
    "rust": ["impl_item"],
    "typescript": ["class_declaration", "interface_declaration"],
}

COMMENT_PREFIX = {
    "python": "#",
    "java": "//",
    "typescript": "//",
    "rust": "//",
    "cpp": "//",
    "go": "//",
}


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


def progress(note: str = "processing"):
    return Progress(
        TextColumn(f"{note} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )


def get_model_name(output_path: str) -> str:
    file_name = Path(output_path).stem
    segments = file_name.split("_")
    output_name = ""
    for segment in segments:
        if segment == "slash":
            output_name += "/"
        else:
            output_name += segment
    return output_name


def save_json(output_json, result_path) -> None:
    if os.path.isfile(result_path):
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(output_json, f)
