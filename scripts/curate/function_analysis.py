import os
from tree_sitter import Parser
from tree_sitter_languages import get_language
from tempfile import TemporaryDirectory
from git import Repo
import json

ts_lang2suffix = { # Changes c++ to cpp
    "python": [".py"],
    "go": [".go"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx", ".c", ".h"],
    "java": [".java"],
    "typescript": [".ts"],
    "php": [".php"],
    "rust": [".rs"],
}


def find_function_definitions(repo_path):
    # Create a parser
    parser = Parser()

    # Supported languages and file extensions
    extension_map = {}
    for language in ts_lang2suffix:
        for suffix in ts_lang2suffix[language]:
            extension_map[suffix] = get_language(language)

    print(extension_map)

    extracted_functions = []

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, file_extension = os.path.splitext(file)

            if file_extension in extension_map:
                language = extension_map[file_extension]
                parser.set_language(language)
                
                with open(file_path, 'r') as f:
                    code = f.read()
                    tree = parser.parse(bytes(code, 'utf8'))

                    query = language.query("""
                        (function_definition
                        name: (identifier) @function.def
                        )
                    """)

                    captures = query.captures(tree.root_node)
                    for capture in captures:
                        function_name = capture[0].text.decode('utf8')
                        line_number = capture[0].start_point[0] + 1
                        extracted_functions.append({
                            "function_name": function_name,
                            "file_path": file_path,
                            "line_number": line_number
                        })
    return extracted_functions
    
if __name__ == "__main__":
    print("Testing function analysis: ")
    print(json.dumps(find_function_definitions(
        "./../../repo/poetry/src/poetry/"
    ), indent=4))