from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from .utils import load_embeddings, load_lookup


class SemanticSearch:
    def __init__(self, logger, embeddings: str, model_name: str):
        self.logger = logger
        self.data = load_embeddings(embeddings)
        self.values = np.array(list(self.data.values()))
        self.keys: np.array[str] = np.array(
            list(self.data.keys())
        )  # these should be strings, check README.md
        self.logger.info("Fitting knn")
        self.tokenizer, self.model = self.setup_model(model_name)

    def _nn_search(self, string, k):
        if k is None:
            k = self.values.shape[0]  # default to all concepts

        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(self.values)
        if string not in self.keys:
            emb = self._get_embeddings(string)
        else:
            emb = self.data[string]

        return [dict(concept=c, distance=d) for c, d in self._get_knn(emb, nbrs)]

    def _get_embeddings(self, string):
        tokens = self.tokenizer(string)["input_ids"]

        output = self.model(torch.tensor([tokens]))

        embeddings = output.last_hidden_state.squeeze()
        embeddings = embeddings[1:-1]  # remove [CLS] and [SEP]
        phrase_embedding = torch.mean(embeddings, dim=0).detach().numpy()

        return phrase_embedding

    def _get_knn(self, emb, nbrs):
        distances, indices = nbrs.kneighbors([emb])
        return [
            (self.keys[i], round(float(d), 3)) for i, d in zip(indices[0], distances[0])
        ]

    def search(self, string, k=None):
        return self._nn_search(string, k)

    def setup_model(self, model_name):
        self.logger.info(f"Setting up model: '{model_name}'")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        return tokenizer, model


class PlainSearch:
    def __init__(self, logger, lookup: str):
        self.logger = logger
        self.df = load_lookup(lookup)

    def search(self, string, k=None):
        results = sorted(
            self._plain_search(string), key=lambda x: x["count"], reverse=True
        )

        if k:
            return results[:k]

        return results

    def _plain_search(self, string):
        return [
            dict(concept=concept, count=count)
            for concept, count in zip(self.df["concept"], self.df["count"])
            if string in concept
        ]
