# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

# Test cases for tree-sitter queries and class/type name fetch analysis

from tree_sitter_languages import get_language, get_parser

from repoqa.utility import FUNCTION_QUERY, topological_sort
from scripts.curate.function_analysis import class_type_analysis


def test_basic_python():
    test_code = """
class TestClass:
    def __init__(self):
        self.data = []

    def do_local():
        spam = "local spam"

class OtherClass:
    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

def do_global():
    global spam
    spam = "global spam"
"""
    fn_query = get_language("python").query(FUNCTION_QUERY["python"])
    parser = get_parser("python")
    code_bytes = bytes(test_code, "utf8")
    tree = parser.parse(code_bytes)
    extracted_classes = []
    for capture in fn_query.captures(tree.root_node):
        node, _ = capture
        function_class_type = class_type_analysis(node, "python")
        print(function_class_type)
        extracted_classes.append(function_class_type)
    assert extracted_classes == ["TestClass", "TestClass", "OtherClass", ""]


def test_basic_java():
    test_code = """
public class Main {
  int x = 5;

  public static void main(String[] args) {
    Main myObj = new Main();
    System.out.println(myObj.x);
  }

  private void other(){
    return 5;
  }
}

class Other{
  private void other(){
    return 5;
  }
}
"""
    lang = "java"
    fn_query = get_language(lang).query(FUNCTION_QUERY[lang])
    parser = get_parser(lang)
    code_bytes = bytes(test_code, "utf8")
    tree = parser.parse(code_bytes)
    extracted_classes = []
    for capture in fn_query.captures(tree.root_node):
        node, _ = capture
        function_class_type = class_type_analysis(node, lang)
        extracted_classes.append(function_class_type)
    assert extracted_classes == ["Main", "Main", "Other"]


def test_basic_cpp():
    test_code = """
struct foo {
  int bar;
  foo() : bar(3) {}   //look, a constructor
  int getBar()
  {
    return bar;
  }
};

class MyClass {        // The class
  public:              // Access specifier
    void myMethod() {  // Method/function defined inside the class
      cout << "Hello World!";
    }
};

int main() {
  MyClass myObj;     // Create an object of MyClass
  myObj.myMethod();  // Call the method
  return 0;
}

"""
    lang = "cpp"
    fn_query = get_language(lang).query(FUNCTION_QUERY[lang])
    parser = get_parser(lang)
    code_bytes = bytes(test_code, "utf8")
    tree = parser.parse(code_bytes)
    extracted_classes = []
    for capture in fn_query.captures(tree.root_node):
        node, _ = capture
        print(node)
        # function_class_type = class_type_analysis(node, lang)
        # extracted_classes.append(function_class_type)
    assert extracted_classes == ["foo", "getBar", "MyClass", ""]
