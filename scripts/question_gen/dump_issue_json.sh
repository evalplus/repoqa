#!/bin/bash

# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

issue_list_file=$(realpath $(dirname "${BASH_SOURCE[0]}")/issue_demo/issue_demo_list.txt)
dumped_issue_json_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/dumped_issue_json)
mkdir -p ${dumped_issue_json_dir}

while read issue_url; do
    array=(${issue_url//\// })
    owner_name=${array[2]}
    repo_name=${array[3]}
    issue_number=${array[5]}

    # echo "Dumping issue json for ${owner_name}/${repo_name}#${issue_number}"
    issue_json_file=${dumped_issue_json_dir}/${owner_name}_${repo_name}_${issue_number}.json
    gh issue view ${issue_number} -R ${owner_name}/${repo_name} --json title,body,createdAt,updatedAt,author,labels,comments > ${issue_json_file}
done < $issue_list_file
