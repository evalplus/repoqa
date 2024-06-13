import os
from typing import List, Tuple

from openai import Client
import numpy as np

from repoqa.provider.embeddings.base import BaseEmbeddingsProvider
from repoqa.provider.request.openai import make_auto_embeddings_request

class OpenAIEmbeddingsProvider(BaseEmbeddingsProvider):
    def __init__(self, model, base_url: str = None):
        self.model = model
        self.client = Client(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=base_url
        )
    
    def find_best_match(
        self, description, snippets, threshold=0
    ) -> str:
        all_texts  = [ description ] + snippets
        embedded_texts = make_auto_embeddings_request(self.client, all_texts, self.model)
        query_embedded = np.array(embedded_texts[0])
        max_similarity = 0
        max_sim_index = 0

        query_norm = np.linalg.norm(query_embedded)

        for i in range(1, len(embedded_texts)):
            similarity_score = (query_embedded @ np.array(embedded_texts[i])) / (query_norm * np.linalg.norm(embedded_texts[i])) # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                max_sim_index = i

        return all_texts[max_sim_index]