from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from search.search import PlainSearch
from search.concept_mentions import ConceptMentions
from predict.predict import Predictor
import logging
from logging.handlers import RotatingFileHandler
import sys
import os
import threading
import time
import uuid
import resource
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()


def _parse_log_level(value, default=logging.INFO):
    if value is None:
        return default

    if isinstance(value, int):
        return value

    value_str = str(value).strip()
    if value_str.isdigit():
        return int(value_str)

    return logging._nameToLevel.get(value_str.upper(), default)


def _env_int(name, default, min_value=None):
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    if min_value is not None and value < min_value:
        return min_value

    return value


def _env_float(name, default, min_value=None):
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError:
        return default

    if min_value is not None and value < min_value:
        return min_value

    return value


def _current_rss_mb():
    try:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(max_rss / (1024 * 1024), 1)
        return round(max_rss / 1024, 1)
    except Exception:
        return None


def setup_logger(file, level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.propagate = False

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    if file:
        directory = os.path.dirname(file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        file_handler = RotatingFileHandler(
            file,
            maxBytes=_env_int("LOG_MAX_BYTES", 10_485_760, min_value=1),
            backupCount=_env_int("LOG_BACKUP_COUNT", 5, min_value=1),
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger(
    os.getenv("LOGS"),
    level=_parse_log_level(os.getenv("LOG_LEVEL"), default=logging.INFO),
    log_to_stdout=True,
)

MAX_CONCEPTS_ITEMS = _env_int("MAX_CONCEPTS_ITEMS", 137_000, min_value=1)
MAX_SEARCH_K = _env_int("MAX_SEARCH_K", 200, min_value=1)
MAX_PREDICT_K = _env_int("MAX_PREDICT_K", 200, min_value=1)
MAX_MENTIONS_K = _env_int("MAX_MENTIONS_K", 20, min_value=1)

MAX_CONCURRENT_PREDICT_REQUESTS = _env_int(
    "MAX_CONCURRENT_PREDICT_REQUESTS", 1, min_value=1
)
MAX_CONCURRENT_MENTIONS_REQUESTS = _env_int(
    "MAX_CONCURRENT_MENTIONS_REQUESTS", 4, min_value=1
)
OVERLOAD_WAIT_SECONDS = _env_float("OVERLOAD_WAIT_SECONDS", 0.1, min_value=0.0)
SLOW_REQUEST_MS = _env_int("SLOW_REQUEST_MS", 1000, min_value=1)
K_MUST_BE_GT_ZERO = "'k' must be greater than 0"

predict_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_PREDICT_REQUESTS)
mentions_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_MENTIONS_REQUESTS)

origins = ["*"]

root_path = os.getenv("ROOT_PATH", "")
app = FastAPI(root_path=root_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            f"request_id={request_id} method={request.method} path={request.url.path} "
            f"status=500 duration_ms={duration_ms:.1f}"
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000
    rss_mb = _current_rss_mb()
    rss_payload = f" rss_max_mb={rss_mb}" if rss_mb is not None else ""

    message = (
        f"request_id={request_id} method={request.method} path={request.url.path} "
        f"status={response.status_code} duration_ms={duration_ms:.1f}{rss_payload}"
    )

    if duration_ms > SLOW_REQUEST_MS:
        logger.warning(message)
    else:
        logger.info(message)

    response.headers["X-Request-ID"] = request_id
    return response


def _run_with_limit(semaphore, operation, func, *args, **kwargs):
    acquired = semaphore.acquire(timeout=OVERLOAD_WAIT_SECONDS)
    if not acquired:
        logger.warning(f"Overload protection triggered for operation='{operation}'")
        raise HTTPException(
            status_code=503,
            detail=f"Server is busy handling {operation} requests. Please retry.",
        )

    try:
        return func(*args, **kwargs)
    finally:
        semaphore.release()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "n/a")
    logger.warning(f"request_id={request_id} validation_error={exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, _exc: MemoryError):
    request_id = getattr(request.state, "request_id", "n/a")
    logger.exception(f"request_id={request_id} memory_error=out_of_memory")
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Server memory pressure detected. Please retry in a few moments.",
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, _exc: Exception):
    request_id = getattr(request.state, "request_id", "n/a")
    logger.exception(f"request_id={request_id} unhandled_exception=true")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


plain_search = PlainSearch(
    logger=logger,
    lookup=os.getenv("LOOKUP"),
)

predictor = Predictor(
    logger=logger,
    lookup=os.getenv("LOOKUP"),
    feature_embeddings=os.getenv("FEATURE_EMBEDDINGS"),
    concept_embeddings=os.getenv("CONCEPT_EMBEDDINGS"),
    adjacency_index=os.getenv("ADJACENCY_INDEX", "data/adjacency"),
    layers=literal_eval(os.getenv("LAYERS")),
    model=literal_eval(os.getenv("MODEL")),
    features=literal_eval(os.getenv("FEATURES")),
    blending=literal_eval(os.getenv("BLENDING")),
)

mentions = None
concept_index_path = os.getenv(
    "CONCEPT_WORK_IDS", "data/lookup/concept_to_work_ids.pkl.gz"
)
works_compact_path = os.getenv(
    "WORKS_COMPACT", "data/misc/works.abstracts.compact.csv.gz"
)

if os.path.exists(concept_index_path) and os.path.exists(works_compact_path):
    mentions = ConceptMentions(
        logger=logger,
        concept_index_path=concept_index_path,
        works_compact_path=works_compact_path,
    )
else:
    logger.warning(
        "Concept mentions index not loaded. Missing files: "
        f"index='{concept_index_path}', works='{works_compact_path}'. "
        "Run 'python pimp_lookup.py' first."
    )

logger.info(
    "Runtime limits: "
    f"MAX_CONCEPTS_ITEMS={MAX_CONCEPTS_ITEMS}, "
    f"MAX_SEARCH_K={MAX_SEARCH_K}, "
    f"MAX_PREDICT_K={MAX_PREDICT_K}, "
    f"MAX_MENTIONS_K={MAX_MENTIONS_K}, "
    f"MAX_CONCURRENT_PREDICT_REQUESTS={MAX_CONCURRENT_PREDICT_REQUESTS}, "
    f"MAX_CONCURRENT_MENTIONS_REQUESTS={MAX_CONCURRENT_MENTIONS_REQUESTS}, "
    f"OVERLOAD_WAIT_SECONDS={OVERLOAD_WAIT_SECONDS}"
)


@app.get("/")
def frontend():
    return FileResponse("frontend.html")


@app.get("/browse")
def browse_frontend():
    return FileResponse("browse.html")


@app.get("/prediction")
def prediction_frontend():
    return FileResponse("prediction.html")


@app.get("/concept-mentions")
def concept_mentions_frontend():
    return FileResponse("mentions.html")


@app.get("/saved-concepts")
def saved_concepts_frontend():
    return FileResponse("saved.html")


@app.get(
    "/concepts",
    responses={422: {"description": "Invalid query parameters"}},
)
def concepts(query: str = "", limit: int = 200, all: bool = False):
    if not all and limit <= 0:
        raise HTTPException(status_code=422, detail="'limit' must be greater than 0")

    df = plain_search.df

    if query:
        filtered = df[df["concept"].str.contains(query, case=False, na=False)]
    else:
        filtered = df

    values = filtered["concept"].dropna().astype(str)
    total = int(len(values))

    if all:
        items = values.tolist()
    else:
        bounded_limit = min(limit, MAX_CONCEPTS_ITEMS)
        items = values.head(bounded_limit).tolist()

    return {
        "total": total,
        "returned": int(len(items)),
        "items": items,
    }


@app.get(
    "/search",
    responses={422: {"description": "Invalid query parameters"}},
)
def search(
    query: str,
    semantic: bool = False,
    k: int = None,
    ignore_case: bool = True,
):
    logger.info(f"Searching term: '{query}'")
    if k is not None:
        if k <= 0:
            raise HTTPException(status_code=422, detail=K_MUST_BE_GT_ZERO)
        k = min(k, MAX_SEARCH_K)

    if semantic:
        logger.warning("Semantic search flag is currently ignored; using plain search")

    return plain_search.search(query, k=k, ignore_case=ignore_case)


@app.get(
    "/predict",
    responses={
        404: {"description": "Concept not found"},
        422: {"description": "Invalid query parameters"},
        503: {"description": "Server overloaded or request rejected for safety"},
    },
)
def predict(concept: str, k: int = 200):
    if k <= 0:
        raise HTTPException(status_code=422, detail=K_MUST_BE_GT_ZERO)
    if k > MAX_PREDICT_K:
        raise HTTPException(
            status_code=422,
            detail=f"'k' too large. Max allowed is {MAX_PREDICT_K}",
        )

    logger.info(f"Predicting for concept: '{concept}'")
    try:
        return _run_with_limit(
            predict_semaphore,
            operation="predict",
            func=predictor.predict,
            concept=concept,
            max_degree=None,
            k=k,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown concept '{concept}'")
    except RuntimeError as exc:
        logger.warning(f"Prediction rejected for concept='{concept}': {exc}")
        raise HTTPException(status_code=503, detail=str(exc))


@app.get(
    "/mentions",
    responses={
        422: {"description": "Invalid query parameters"},
        503: {"description": "Mentions index not available on this backend instance"},
    },
)
def concept_mentions(concept: str, k: int = 10):
    logger.info(f"Loading mentions for concept: '{concept}'")

    if k <= 0:
        raise HTTPException(status_code=422, detail=K_MUST_BE_GT_ZERO)
    k = min(k, MAX_MENTIONS_K)

    if mentions is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Mentions index unavailable. Build it via 'python pimp_lookup.py' and "
                "set CONCEPT_WORK_IDS / WORKS_COMPACT if needed."
            ),
        )

    return _run_with_limit(
        mentions_semaphore,
        operation="mentions",
        func=mentions.get_mentions,
        concept=concept,
        k=k,
    )
