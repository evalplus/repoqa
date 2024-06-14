from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingsProvider(ABC):
    @abstractmethod
    def find_best_match(
        self, description, snippets, threshold=0
    ) -> str:
        ...