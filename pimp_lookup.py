import argparse
import gzip
import json
import pickle
from ast import literal_eval

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build concept mentions indices from works and lookup files"
    )
    parser.add_argument(
        "--works",
        default="data/materials-science.llama2.works.csv",
        help="Path to works CSV",
    )
    parser.add_argument(
        "--lookup",
        default="data/lookup/lookup.M.csv",
        help="Path to base lookup CSV",
    )
    parser.add_argument(
        "--output-lookup-works",
        default="data/lookup/lookup.M.works.csv",
        help="Output CSV path for lookup with works column",
    )
    parser.add_argument(
        "--output-concept-index",
        default="data/lookup/concept_to_work_ids.pkl.gz",
        help="Output compressed pickle path for concept -> [work_ids]",
    )
    parser.add_argument(
        "--output-works-compact",
        default="data/misc/works.abstracts.compact.csv.gz",
        help="Output compressed CSV path for compact works payload",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="CSV chunksize used for streaming the works file",
    )
    return parser.parse_args()


def safe_parse_concepts(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []

    try:
        parsed = literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, (set, tuple)):
            return list(parsed)
        if isinstance(parsed, str):
            return [parsed]
        return []
    except (ValueError, SyntaxError):
        return []


def _split_element_tokens(items):
    elements = []
    for item in items:
        for part in str(item).split(","):
            token = part.strip().strip('"').strip("'")
            if token:
                elements.append(token)
    return elements


def _normalize_elements_input(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    raw = str(value).strip()
    if not raw:
        return []

    parsed = raw
    if raw[0] in "[{\"'" and raw[-1] in "]}\"'":
        try:
            parsed = literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = raw

    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def parse_elements(value):
    return _split_element_tokens(_normalize_elements_input(value))


def main():
    args = parse_args()

    lookup = pd.read_csv(args.lookup)
    lookup_concepts = set(lookup["concept"].dropna().astype(str).tolist())

    concept_works_map = {concept: [] for concept in lookup_concepts}
    selected_work_payload = {}

    iterator = pd.read_csv(
        args.works,
        usecols=["id", "doi", "abstract", "llama_concepts", "elements"],
        chunksize=args.chunksize,
    )

    for chunk in tqdm(iterator, desc="Indexing works"):
        chunk["id"] = chunk["id"].astype(str)
        chunk["doi"] = chunk["doi"].fillna("").astype(str)
        chunk["abstract"] = chunk["abstract"].fillna("").astype(str)

        for row in chunk.itertuples(index=False):
            work_id = row.id
            concepts = {
                str(concept)
                for concept in (
                    safe_parse_concepts(row.llama_concepts)
                    + parse_elements(row.elements)
                )
            }

            matched = False
            for concept in concepts:
                if concept in lookup_concepts:
                    concept_works_map[concept].append(work_id)
                    matched = True

            if matched and work_id not in selected_work_payload:
                selected_work_payload[work_id] = {
                    "id": work_id,
                    "doi": row.doi,
                    "abstract": row.abstract,
                }

    for concept, work_ids in concept_works_map.items():
        concept_works_map[concept] = sorted(set(work_ids))

    lookup["works"] = (
        lookup["concept"]
        .astype(str)
        .map(lambda c: json.dumps(concept_works_map.get(c, []), ensure_ascii=False))
    )
    lookup.to_csv(args.output_lookup_works, index=False)

    with gzip.open(args.output_concept_index, "wb") as f:
        pickle.dump(concept_works_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    compact_df = pd.DataFrame(selected_work_payload.values())
    compact_df.to_csv(args.output_works_compact, index=False, compression="gzip")


if __name__ == "__main__":
    main()
