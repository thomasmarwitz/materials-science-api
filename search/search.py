from transformers import AutoTokenizer, AutoModel
import pickle
import gzip
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import pandas as pd


def score(distance):
    if distance < 0:
        raise ValueError("Distance must be a non-negative number.")

    elif distance <= 10:
        return 100 - ((distance - 0) / (10 - 0)) * (100 - 90)

    elif distance <= 15:
        return 90 - ((distance - 10) / (15 - 10)) * (90 - 60)

    elif distance <= 20:
        return 60 - ((distance - 15) / (20 - 15)) * (60 - 20)

    elif distance <= 30:
        return 20 - ((distance - 20) / (30 - 20)) * (20 - 5)

    else:
        return max(
            5 - ((distance - 30) / (100 - 30)) * 5, 0
        )  # The 100 here is a presumed upper limit for interpolation


def setup_model(model_name):
    print("Setting up model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


def convert_tensors_to_arrays(data):
    return {k: v.numpy() for k, v in data.items()}


class SemanticSearch:
    def __init__(self, data, model_name):
        self.data = data
        self.values = np.array(list(self.data.values()))
        self.keys = np.array(list(self.data.keys()))
        print("Fitting knn")
        self.tokenizer, self.model = setup_model(model_name)

    def _nn_search(self, string, k):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(self.values)
        if string not in self.keys:
            emb = self._get_embeddings(string)
        else:
            emb = self.data[string]

        return [
            dict(match=c, score=round(score(d), 1), distance=d)
            for c, d in self._get_knn(emb, nbrs)
        ]

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

    def search(self, string, k=10):
        return self._nn_search(string, k)


class PlainSearch:
    def __init__(self, df):
        self.df = df

    def search(self, string, k=10):
        return sorted(
            self._plain_search(string), key=lambda x: x["count"], reverse=True
        )[:k]

    def _plain_search(self, string):
        return [
            dict(match=concept, count=count)
            for concept, count in zip(self.df["concept"], self.df["count"])
            if string in concept
        ]


print("Loading data")
CONCEPT_PATH = "data/embeddings/embeddings.full.M.pkl.gz"
data = convert_tensors_to_arrays(load_compressed(CONCEPT_PATH))

sem_search = SemanticSearch(
    data,
    model_name="m3rg-iitd/matscibert",
)


plain_search = PlainSearch(pd.read_csv("data/lookup/lookup_medium.csv"))

# semantic relevance score
# 0 - 5: 100-95
# 5-10: 95-80
# 10-15: 80-40
# 15-20: 40-20
# 20-30: 20-5
# 30-inf: 5-0
