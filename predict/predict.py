import torch
import numpy as np

from graph import Graph
from net import Network
from search.utils import load_compressed, load_lookup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class Predictor:
    def __init__(
        self,
        logger,
        lookup: str,
        feature_embeddings: str,
        concept_embeddings: str,
        graph: str,
        since: int,
        layers: str,
        model: str,
    ):
        self.logger = logger

        logger.info(f"Loading feature embeddings from '{feature_embeddings}'")
        self.feature_embeddings = load_compressed(feature_embeddings)

        logger.info(f"Loading concept embeddings from '{concept_embeddings}'")
        self.concept_embeddings = load_compressed(concept_embeddings)

        logger.info(f"Loading lookup from '{lookup}'")
        self.lookup = load_lookup(lookup)
        self.lookup_c_id = {
            concept: id for concept, id in zip(lookup["concept"], lookup["id"])
        }
        self.lookup_id_c = {
            id: concept for concept, id in zip(lookup["concept"], lookup["id"])
        }

        logger.info(f"Loading model from '{model}' with layers '{layers}'")
        self.model = Predictor.load_model(layers, path=model)

        logger.info(f"Loading graph from '{graph}'")
        _graph = Graph.from_path(graph)
        self.g = Graph.from_edge_list(_graph.get_until_year(since))
        self.g_nx = _graph.get_nx_graph(since)

    def predict(
        self, concept: str, max_degree: int = None, max_depth: int = None, k: int = 10
    ):
        concept_id = self.lookup_c_id[concept]

        pairs = self.get_pairs(concept_id, max_degree, max_depth)

        inputs = self.get_embeddings(pairs).to(device)

        outs = self.model(inputs)
        outs = outs.detach().cpu().numpy().flatten()

        results = [(self.lookup_id_c[id], score) for id, score in zip(pairs, outs)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def get_pairs(self, concept_id, max_degree=None, max_depth=None):
        unconnected = []

        for other in self.g.vertices:
            if other == concept_id:
                continue

            if max_degree is not None and self.g.degree[other] > max_degree:
                continue

            if self.g_nx.has_edge(concept_id, other):
                continue

            unconnected.append(other)

        return torch.tensor([(concept_id, other) for other in unconnected])

    def get_embeddings(self, pairs):
        l = []
        for v1, v2 in pairs:
            i1 = int(v1.item())
            i2 = int(v2.item())

            emb1_f = np.array(self.feature_embeddings[i1])
            emb2_f = np.array(self.feature_embeddings[i2])

            emb1_c = np.array(self.concept_embeddings[i1])
            emb2_c = np.array(self.concept_embeddings[i2])

            l.append(np.concatenate([emb1_f, emb2_f, emb1_c, emb2_c]))
        return torch.tensor(np.array(l)).float()

    @staticmethod
    def load_model(layers: str, path: str):
        layers = layers.split(",")
        model = Network(layers).to(device)

        model.load_state_dict(
            torch.load(
                path,
                map_location=torch.device(device),
            )
        )

        return model