# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

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
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
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
