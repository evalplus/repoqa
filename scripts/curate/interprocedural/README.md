## Interprocedural Analysis

### Problem Description
Context & Flow Insensitive Interprocedural Path Finding: Given a Function A and another function B (preferrably in a different file) find the shortest path between function A and function B. That is, the shortest path in the call graph that can go from function to function B. We can also have negative cases where no such path exist and the LLM has to return so. We use the term context & flow insensitive path here because we don't consider whether it is actually reachable in execution (since for these repos, the inputs are very complex), only use static analysis.


### Construction Steps
Step 1: For each function, find the files that it is in. Think
about cases with duplicate functions where a function name appears in multiple files

Step 2: For each repo, construct interprocedural dependency graph
  - Consideration: Should we consider some more indepth control
  flow graph where we look at specific call site location with function as well instead of
  simply just function A calls function B

Format for interprocedural dependency graph. The call graph should be stored as a json
```json
{
  full_name: {
    file_name: str,
    class_name: str,
    function_name: str,
    is_unique: bool,
    targets: [str],
  },
  ...
}
```
- `full_name`: Full name of the function, which has the format `file::class.function_name` (Note that for Rust and Go, the class refers to type)
- `file_name`: Name of the file where this function is located
- `is_unique`: Is there a copy of the function with the exact same full name
- `class_name`: Name of the class where this function is located (empty string if there is string )
- `function_name`: Name of the function
- `targets`: list of targets that this function calls

Step 3: Choose a function, traverse call graph and find another function in a different file (with a path length greater than X, can be set later). Construct long context that contains all functions within the path and join them together to construct long file. The path should only contain functions with unique names atm.
