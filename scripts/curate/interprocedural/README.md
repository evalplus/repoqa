## Interprocedural Analysis

### Problem Description
Realizable Interprocedural Path Finding: Given a Function A and another function B (preferrably in a different file) find the shortest realizable path between function A and function B. That is, the shortest path in the call graph that can go from function to function B. We can also have negative cases where no such path exist and the LLM has to return so. We use the term realizable path here because we don't consider whether it is actually reachable in execution (since for these repos, the inputs are very complex), only use static analysis.


### Construction Steps
- Step 1: For each function, find the files that it is in. Think
about cases with duplicate functions where a function name appears in multiple files

- Step 2: For each repo, construct interprocedural dependency graph
  - Consideration: Should we consider some more indepth control
  flow graph where we look at specific call site location with function as well instead of
  simply just function A calls function B

- Step 3: Choose a function, traverse call graph and find another function in a different file (with a path length greater than X, can be set later). Construct long context that contains all functions within the path and join them together to construct long file. The path should only contain functions with unique names atm.
