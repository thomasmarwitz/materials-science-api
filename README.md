## Start the server

`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

## Create modified lookup.csv

Add extra column `works` to `lookup.M.csv` (save it as `lookup.M.works.csv`) by running `python pimp_lookup.py`.

## Embeddings for Semantic Search

The embeddings file (located at `data/embeddings/semantic`) powering the semantic search should have **strings** as keys. When using the _average embeddings_ script, make sure to set `--store_concept_ids False`
