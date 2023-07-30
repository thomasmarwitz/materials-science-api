import pandas as pd
import pickle
import gzip


def load_compressed(path):
    with open(path, "rb") as f:
        compressed = f.read()
    return pickle.loads(gzip.decompress(compressed))


def convert_tensors_to_arrays(data):
    return {k: v.numpy() for k, v in data.items()}


def load_embeddings(concept_path):
    return convert_tensors_to_arrays(load_compressed(concept_path))


def load_lookup(lookup_path):
    return pd.read_csv(lookup_path)
