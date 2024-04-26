# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

lang2suffix = {
    "python": [".py"],
    "go": [".go"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx", ".c", ".h"],
    "java": [".java"],
    "typescript": [".ts", ".js"],
    "php": [".php"],
    "rust": [".rs"],
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

FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
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
    """
}