"""
Make it harder for an LLM to predict code functionality based on comments or function names.
"""
import re
from tree_sitter_languages import get_language, get_parser
from utility import COMMENT_QUERY, FUNCTION_NAME_QUERY
import json
from tqdm import tqdm

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
    print(f"Running rename_functions: {code}, {language}")
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

def main(
    dataset_filepath: str
):
    dataset_file = open(dataset_filepath, "r")
    dataset = dataset_file.read()
    dataset_file.close()
    dataset = json.loads(dataset)
    for lang in dataset:
        print(f"Loading language: {lang}")
        for repo_idx in range(len(dataset[lang])):
            print(f"Loading repo: {dataset[lang][repo_idx]["repo"]}")
            for file in tqdm(dataset[lang][repo_idx]["content"]):
                content = dataset[lang][repo_idx]["content"][file]
                if len(content.strip()) == 0:
                    continue
                dataset[lang][repo_idx]["content"][file] = remove_comments(content, lang)
    path_no_ext = ".".join(dataset_filepath.split(".")[:-1])
    
    new_filepath = f"{path_no_ext}-no-comments.json"
    gen_dataset_file = open(new_filepath, "w+")
    gen_dataset_file.write(json.dumps(dataset))
    gen_dataset_file.close()
    print(f"Saved to {new_filepath}")

if __name__ == "__main__":
    from fire import Fire
    Fire(main)