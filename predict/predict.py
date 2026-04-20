import torch
import numpy as np
from typing import Optional
from ast import literal_eval

from .adjacency import AdjacencyIndex
from .net import Network
from .utils import load_compressed, load_lookup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class Model:
    def __init__(self, model):
        self._model = model

    def __call__(self, inputs):
        return self._model(inputs).detach().cpu().numpy().flatten()


class Mixture:
    def __init__(self, models, blending):
        self.models = models
        self.blending = blending

    def __call__(self, inputs):
        outputs = [
            model(_input) * factor
            for model, _input, factor in zip(self.models, inputs, self.blending)
        ]

        return np.sum(outputs, axis=0)


class Predictor:
    def __init__(
        self,
        logger,
        lookup: str,
        feature_embeddings: str,
        concept_embeddings: str,
        adjacency_index: str,
        layers: list[str],
        model: list[str],
        features: list[str],
        blending: list[float] = None,
    ):
        self.logger = logger

        logger.info(f"Loading feature embeddings from '{feature_embeddings}'")
        self.feature_embeddings = load_compressed(feature_embeddings)[
            "v_features"
        ]  # new data format

        logger.info(f"Loading concept embeddings from '{concept_embeddings}'")
        self.concept_embeddings = Predictor.transform_to_array(
            load_compressed(concept_embeddings)
        )

        logger.info(f"Loading lookup from '{lookup}'")
        self.lookup = load_lookup(lookup)
        self.lookup_c_id = dict(zip(self.lookup["concept"], self.lookup["id"]))
        self.lookup_id_c = dict(zip(self.lookup["id"], self.lookup["concept"]))

        logger.info(f"Loading model from '{model}' with layers '{layers}'")
        self.model = Predictor.load_model(layers, path=model, blending=blending)
        self.features = [
            [literal_eval(choice) for choice in string.split(",")]
            for string in features
        ]
        logger.info(f"Using features: {self.features}")

        logger.info(f"Loading adjacency index from '{adjacency_index}'")
        self.adjacency = AdjacencyIndex(adjacency_index)

    def predict(
        self, concept: str, max_degree: Optional[int] = None, k: Optional[int] = 10
    ):
        concept_id = self.lookup_c_id[concept]
        pairs = self._get_pairs(concept_id, max_degree)
        inputs = self._get_embeddings(pairs, features=self.features)
        outs = self._predict(inputs)
        results = self._create_response(outs, pairs, k)
        return results

    def _get_pairs(self, concept_id, max_degree: Optional[int] = None):
        self.logger.debug("Getting pairs")
        unconnected = []

        vertices = self.adjacency.vertices
        neighbors = self.adjacency.neighbor_set(concept_id)

        for other in vertices:
            other = int(other)

            if other == concept_id:
                continue

            if max_degree is not None and self.adjacency.degree(other) > max_degree:
                continue

            if other in neighbors:
                continue

            unconnected.append(other)

        pairs = torch.tensor([(concept_id, other) for other in unconnected])
        self.logger.debug(f"Got {len(pairs)} pairs")
        return pairs

    def _get_embeddings(self, pairs, features):
        if len(features) == 1:
            return self._retrieve_embeddings(pairs, features[0])

        return [
            self._retrieve_embeddings(pairs, feature_choice)
            for feature_choice in features
        ]

    def _retrieve_embeddings(self, pairs, feature_choice):
        self.logger.debug(
            f"Getting embeddings for: {len(pairs)} pairs with feature choice: {feature_choice}"
        )
        l = []
        for v1, v2 in pairs:
            i1 = int(v1.item())
            i2 = int(v2.item())

            feature_list = []

            if feature_choice[0]:
                emb1_f = self.feature_embeddings[i1]
                emb2_f = self.feature_embeddings[i2]
                feature_list.extend([emb1_f, emb2_f])

            if feature_choice[1]:
                emb1_c = self.concept_embeddings[i1]
                emb2_c = self.concept_embeddings[i1]
                feature_list.extend([emb1_c, emb2_c])

            assert len(feature_list) > 0
            l.append(np.concatenate(feature_list))
        return torch.tensor(np.array(l)).float().to(device)

    def _predict(self, inputs):
        self.logger.debug("Predicting")
        outs = self.model(inputs)
        return outs

    def _create_response(self, outs, pairs, k: Optional[int]):
        self.logger.debug(f"Creating response of {k} data points")
        sorted_indices = np.argsort(outs)[::-1]

        top_k_indices = sorted_indices[:k]

        return [
            {"concept": self.lookup_id_c[pairs[i][1].item()], "score": float(outs[i])}
            for i in top_k_indices
        ]

    @staticmethod
    def load_model(layers: list[str], path: list[str], blending: list[float] = None):
        if len(path) == 1:
            return Predictor.load_single_model(layers[0], path[0])

        models = [
            Predictor.load_single_model(layer, p) for layer, p in zip(layers, path)
        ]

        return Mixture(models, blending=blending)

    @staticmethod
    def load_single_model(layers, path):
        layers = [int(l) for l in layers.split(",")]
        model = Network(layers).to(device)

        model.load_state_dict(
            torch.load(
                path,
                map_location=torch.device(device),
            )
        )

        return Model(model)

    @staticmethod
    def transform_to_array(data: dict):
        length = max(data.keys()) + 1

        # 2D array of shape (length, 768)
        array = np.zeros((length, 768))

        for key, value in data.items():
            array[key] = np.array(value)

        return array
