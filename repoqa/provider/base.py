# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List

SYSTEM_MSG = "You are a helpful assistant good at code understanding."


class BaseProvider(ABC):
    @abstractmethod
    def generate_reply(
        self, question, n=1, max_tokens=1024, temperature=0
    ) -> List[str]:
        ...
