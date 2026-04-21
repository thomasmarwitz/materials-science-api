import gzip
import pickle

import pandas as pd


class ConceptMentions:
    def __init__(self, logger, concept_index_path: str, works_compact_path: str):
        self.logger = logger
        self.concept_index_path = concept_index_path
        self.works_compact_path = works_compact_path

        self.concept_to_work_ids = self._load_concept_index(concept_index_path)
        self.works_payload = self._load_works_payload(works_compact_path)

        self.logger.info(
            f"Concept mentions index loaded: {len(self.concept_to_work_ids)} concepts, "
            f"{len(self.works_payload)} works"
        )

    def _load_concept_index(self, path):
        with gzip.open(path, "rb") as file_handle:
            index = pickle.load(file_handle)

        return {
            str(concept): [str(work_id) for work_id in work_ids]
            for concept, work_ids in index.items()
        }

    def _load_works_payload(self, path):
        df = pd.read_csv(path, usecols=["id", "doi", "abstract"])
        df["id"] = df["id"].astype(str)
        df["doi"] = df["doi"].fillna("").astype(str)
        df["abstract"] = df["abstract"].fillna("").astype(str)

        return {
            row.id: {"id": row.id, "doi": row.doi, "abstract": row.abstract}
            for row in df.itertuples(index=False)
        }

    def get_mentions(self, concept: str, k: int = 10):
        work_ids = self.concept_to_work_ids.get(concept, [])
        if k is not None:
            work_ids = work_ids[:k]

        items = [
            self.works_payload[work_id]
            for work_id in work_ids
            if work_id in self.works_payload
        ]

        return {
            "concept": concept,
            "total": len(work_ids),
            "items": items,
        }
