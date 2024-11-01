# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0
import json
from collections import deque
from typing import Dict, Set, Tuple

from tqdm.auto import tqdm
from transformers import AutoTokenizer

TASKS_PER_REPO = 10


class PathFunction:
    def __init__(
        self,
        name: str,
        function_distance: int,
        file_distance: int,
        file_set: Set[str],
        token_size: int,
        cur_file: str,
        cur_path: str,
    ):
        self.name = name
        self.function_distance = function_distance
        self.file_distance = file_distance
        # TODO: Inefficient and not scalable
        self.file_set = set(file_set)
        self.file_set.add(cur_file)
        self.cur_path = list(cur_path)
        self.token_size = token_size
        self.cur_path.append(name)


def find_potential_pair(
    start: str,
    call_graph: Dict[str, Dict],
    function_distance: int,
    file_distance: int,
    max_tokens: int,
    file_content: Dict[str, str],
) -> Tuple[str, str]:
    """
    find_potential_pair: Finds potential pairs of functions
    with path distance of at least function_distance and file distance
    of at least file_distance. Follows the following algorithm

    1. Initiate BFS algorithm
    2. Pop Start of Queue
    3. If function meets criteria return it
    4. Check the neighbours of current function
    5. If any exceed context window length, discard the content
    6. Add any non-visited neighbours into the queue
    """
    output = []
    # Use Llama tokenizer to approximate number of context tokens
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    visited = set()
    start_file_content = file_content[call_graph[start]["file"]]
    queue = deque(
        [
            PathFunction(
                name=start,
                function_distance=0,
                file_distance=0,
                file_set=set(),
                token_size=len(tokenizer.tokenize(start_file_content)),
                cur_file=call_graph[start]["file"],
                cur_path=[],
            )
        ]
    )

    while len(queue):
        current_func = queue.popleft()
        if current_func.token_size > max_tokens:
            continue

        if (
            current_func.function_distance >= function_distance
            and current_func.file_distance >= file_distance
        ):
            return {
                "start": start,
                "end": current_func.name,
                "path": current_func.cur_path,
                "files": list(current_func.file_set),
            }
        targets = call_graph[current_func.name]["targets"]
        for target in targets:
            file = call_graph[target]["file"]

            token_size = current_func.token_size
            if not (file in visited):
                token_size += len(tokenizer.tokenize(file_content[file]))

            if not (target in visited):
                visited.add(target)
                queue.append(
                    PathFunction(
                        name=target,
                        function_distance=current_func.function_distance + 1,
                        file_distance=current_func.function_distance + 1
                        if not file in current_func.file_set
                        else current_func.file_distance,
                        file_set=current_func.file_set,
                        token_size=token_size,
                        cur_file=file,
                        cur_path=current_func.cur_path,
                    )
                )
    return output


def main(
    dataset_path: str,
    function_distance: int = 3,
    file_distance: int = 2,
    max_tokens: int = 16 * 1024,
):
    with open(dataset_path) as f:
        dataset = json.load(f)

    for lang, repos in dataset.items():
        # TODO: remove
        if lang != "python":
            continue

        for repo in tqdm(repos):
            valid_pairs = []
            if not "call_graph" in repo:
                continue
            call_graph = repo["call_graph"]
            content = repo["content"]
            for function in call_graph:
                result = find_potential_pair(
                    function,
                    call_graph,
                    function_distance,
                    file_distance,
                    max_tokens,
                    content,
                )
                if result:
                    valid_pairs.append(result)
            repo["interprocedural_pairs"] = valid_pairs
            print(f"🎉 Found {len(valid_pairs)} pairs in {repo['repo']} ({lang})")

    with open(dataset_path, "w") as f_out:
        json.dump(dataset, f_out)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
