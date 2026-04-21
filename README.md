## Start the server

`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

Before starting the server, build the adjacency index once:

`pixi run build-adjacency`

Set `ADJACENCY_INDEX` in your environment to the output directory (for example `data/adjacency`).

The backend exposes `/search` and `/predict` endpoints. `/generate_abstracts` was removed.

## Build concept mentions index

To support concept -> work abstract lookup (`/mentions` endpoint), build the mentions index once:

`pixi run build-mentions-index`

This generates:

- `data/lookup/lookup.M.works.csv` (lookup with `works` as JSON array of work ids)
- `data/lookup/concept_to_work_ids.pkl.gz` (compressed concept -> work ids index)
- `data/misc/works.abstracts.compact.csv.gz` (compressed compact works payload with `id`, `doi`, `abstract`)

Optional environment variables:

- `LOOKUP_WORKS` (default: `data/lookup/lookup.M.works.csv`)
- `WORKS_COMPACT` (default: `data/misc/works.abstracts.compact.csv.gz`)

Endpoint:

- `GET /mentions?concept=<concept>&k=<max_results>`

## Create modified lookup.csv

Add extra column `works` to `lookup.M.csv` (save it as `lookup.M.works.csv`) by running `python pimp_lookup.py`.

## Embeddings for Semantic Search

The embeddings file (located at `data/embeddings/semantic`) powering the semantic search should have **strings** as keys. When using the _average embeddings_ script, make sure to set `--store_concept_ids False`
