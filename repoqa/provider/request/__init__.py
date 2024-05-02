# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0


def construct_message_list(message, system_message=None):
    msglist = [{"role": "user", "content": message}]
    if system_message:
        msglist.insert(0, {"role": "system", "content": system_message})
    return msglist


def hacky_assistant_stop_seq(tokenizer) -> str:
    _magic_string_ = "&==NowOrNever==&Accelerate!!!==&"
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": _magic_string_},
        ],
        tokenize=False,
    ).split(_magic_string_)[-1]
