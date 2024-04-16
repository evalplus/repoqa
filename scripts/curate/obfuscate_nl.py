"""
Make it harder for an LLM to predict code functionality based on comments or function names.
"""
import re
from tree_sitter_languages import get_language, get_parser
from utility import COMMENT_QUERY, FUNCTION_NAME_QUERY
import json

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
        code = code.replace(chunk, "") #TODO: How do you deal with empty commpents such as a single "#" line?
    return code

def rename_functions(code, language, starting_index=0):
    func_name_query = get_language(language).query(FUNCTION_NAME_QUERY[language])
    parser = get_parser(language)
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)
    function_names = set()
    for capture in func_name_query.captures(tree.root_node):
        node, _  = capture
        function_names.add(node.text.decode("utf-8"))
    function_map = {}
    current_index = starting_index
    for name in function_names:
        function_map[name] = f"function_{starting_index}"
        code = code.replace(name, function_map[name])
        current_index += 1
    return code, function_map

def main():
    test_file = open("function_selection.py", "r")
    code = test_file.read()
    test_file.close()
    code = remove_comments(code, "python")
    code, func_map = rename_functions(code, "python")
    print(code)
    print(json.dumps(func_map))

if __name__ == "__main__":
    from fire import Fire
    Fire(main)