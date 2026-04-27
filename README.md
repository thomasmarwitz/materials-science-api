## Start the server

`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

For production (no auto-reload):

`uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 5`

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

- `CONCEPT_WORK_IDS` (default: `data/lookup/concept_to_work_ids.pkl.gz`)
- `WORKS_COMPACT` (default: `data/misc/works.abstracts.compact.csv.gz`)

Endpoint:

- `GET /mentions?concept=<concept>&k=<max_results>`

## Runtime hardening and logging

The backend now includes request-level logging, overload protection, and bounded caches.

Key environment variables:

- `LOG_MAX_BYTES` (default: `10485760`) and `LOG_BACKUP_COUNT` (default: `5`) for rotating log files
- `SLOW_REQUEST_MS` (default: `1000`) to mark slow requests as warning
- `MAX_CONCEPTS_ITEMS` (default: `500`) max items returned by `/concepts`
- `MAX_SEARCH_K` (default: `200`) max `k` accepted by `/search`
- `MAX_PREDICT_K` (default: `200`) max `k` accepted by `/predict`
- `MAX_MENTIONS_K` (default: `20`) max `k` accepted by `/mentions`
- `MAX_CONCURRENT_PREDICT_REQUESTS` (default: `1`) parallel `predict` requests
- `MAX_CONCURRENT_MENTIONS_REQUESTS` (default: `4`) parallel `mentions` requests
- `OVERLOAD_WAIT_SECONDS` (default: `0.1`) queue wait before returning `503` under load

Predictor cache/memory controls:

- `PREDICT_PAIRS_CACHE_SIZE` (default: `32`)
- `PREDICT_RESULTS_CACHE_SIZE` (default: `64`)
- `PREDICT_MAX_CACHED_RESULTS_PER_CONCEPT` (default: `512`)
- `PREDICT_MAX_PAIRS_TO_CACHE` (default: `100000`)
- `PREDICT_MAX_PAIRS_PER_REQUEST` (default: `500000`)

## Create modified lookup.csv

Add extra column `works` to `lookup.M.csv` (save it as `lookup.M.works.csv`) by running `python pimp_lookup.py`.

## Embeddings for Semantic Search

The embeddings file (located at `data/embeddings/semantic`) powering the semantic search should have **strings** as keys. When using the _average embeddings_ script, make sure to set `--store_concept_ids False`
