import os
from tree_sitter import Parser
from tree_sitter_languages import get_language
from utility import lang2suffix
from tempfile import TemporaryDirectory
from git import Repo

def find_function_definitions(repo_path):
    # Create a parser
    parser = Parser()

    # Supported languages and file extensions
    extension_map = {}
    for language in lang2suffix:
        for suffix in lang2suffix[language]:
            extension_map[suffix] = get_language(language)

    extracted_functions = []

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, file_extension = os.path.splitext(file)

            if file_extension in extension_map:
                parser.set_language(extension_map[file_extension])

                with open(file_path, 'r') as f:
                    code = f.read()
                    tree = parser.parse(bytes(code, 'utf8'))

                    query = language.query("""
                        (function_definition
                            name: (identifier) @function_name
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

def get_functions_from_repo(repo_url, entry_dir):
    with TemporaryDirectory() as tmpdir:
        Repo.clone_from(repo_url, tmpdir)
        return find_function_definitions(os.path.join(tmpdir, entry_dir))
        


# Specify the path to your repository
repo_path = '/path/to/your/repo'

find_function_definitions(repo_path)
