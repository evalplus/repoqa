import os
from typing import List, Tuple

from openai import Client
from sklearn.metrics.pairwise import cosine_similarity
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
        similarities = cosine_similarity([embedded_texts[0]], embedded_texts[1:])[0]
        index = np.argmax(similarities)
        return all_texts[index + 1]