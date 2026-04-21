import json
from ast import literal_eval

import pandas as pd


class ConceptMentions:
    def __init__(self, logger, lookup_works_path: str, works_compact_path: str):
        self.logger = logger
        self.lookup_works_path = lookup_works_path
        self.works_compact_path = works_compact_path

        self.concept_to_work_ids = self._load_concept_index(lookup_works_path)
        self.works_payload = self._load_works_payload(works_compact_path)

        self.logger.info(
            f"Concept mentions index loaded: {len(self.concept_to_work_ids)} concepts, "
            f"{len(self.works_payload)} works"
        )

    def _parse_works(self, value):
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if not isinstance(value, str):
            return []

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = literal_eval(value)
            except (ValueError, SyntaxError):
                return []

        if isinstance(parsed, (list, set, tuple)):
            return [str(v) for v in parsed]

        return []

    def _load_concept_index(self, path):
        df = pd.read_csv(path, usecols=["concept", "works"])
        return {
            str(concept): self._parse_works(works)
            for concept, works in zip(df["concept"], df["works"])
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
