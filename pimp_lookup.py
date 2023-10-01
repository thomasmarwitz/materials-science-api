import pandas as pd
from ast import literal_eval
from tqdm import tqdm

df = pd.read_csv("data-v2/misc/materials-science.llama2.works.csv")

df = df[["id", "llama_concepts"]]
df.llama_concepts = df.llama_concepts.apply(literal_eval)

lookup = pd.read_csv("data-v2/lookup/lookup.M.csv")
to_consider = set(lookup.concept.tolist())


concept_works_map = {}
for id, concepts in tqdm(zip(df.id, df.llama_concepts), total=len(df)):
    for concept in concepts:
        if concept not in to_consider:
            continue

        if concept not in concept_works_map:
            concept_works_map[concept] = set()

        concept_works_map[concept].add(id)

works = pd.DataFrame(concept_works_map.items(), columns=["concept", "works"])

lookup = lookup.merge(works, on="concept", how="left")

lookup.to_csv("data-v2/lookup/lookup.M.works.csv", index=False)
