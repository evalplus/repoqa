# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0


def construct_message_list(message, system_message=None):
    msglist = [{"role": "user", "content": message}]
    if system_message:
        msglist.insert(0, {"role": "system", "content": system_message})
    return msglist
