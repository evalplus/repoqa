# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from scripts.question_gen import ISSUE_SEP

QUESTIONER_ROLE = "QUESTIONER"


def format_issue_dialogue(question_pair, comment_tuples):
    question_author, question = question_pair
    dialogue = f"{question_author}({QUESTIONER_ROLE}) asks: {question}"
    dialogue += ISSUE_SEP
    for comment_author, comment_author_role, comment in comment_tuples:
        if comment_author == question_author:
            comment_author_role = QUESTIONER_ROLE
        dialogue += f"\n\n{comment_author}({comment_author_role}) replies: {comment}"
    return dialogue


def make_readable_issue_from_json(issue_json_file):
    with open(issue_json_file, "r") as f:
        data = json.load(f)
    question = data["body"]
    question_author = data["author"]["login"]
    question_pair = (question_author, question)
    comment_tuples = []
    for comment in data["comments"]:
        comment_tuples.append(
            (comment["author"]["login"], comment["authorAssociation"], comment["body"])
        )

    return format_issue_dialogue(question_pair, comment_tuples)


if __name__ == "__main__":
    dumped_issue_json_dir = Path(__file__).parent / "dumped_issue_json"
    issue_dialogue_dir = Path(__file__).parent / "issue_dialogue"
    issue_dialogue_dir.mkdir(exist_ok=True)
    for issue_json_file in dumped_issue_json_dir.iterdir():
        issue_dialogue = make_readable_issue_from_json(issue_json_file)
        issue_dialogue_file = issue_dialogue_dir / (issue_json_file.stem + ".txt")
        with open(issue_dialogue_file, "w") as f:
            f.write(issue_dialogue)
