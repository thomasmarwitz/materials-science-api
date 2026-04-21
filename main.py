from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from search.search import PlainSearch
from search.concept_mentions import ConceptMentions
from predict.predict import Predictor
import logging
import sys
import os
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()


def setup_logger(file, level=logging.INFO, log_to_stdout=True):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    if log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger(
    os.getenv("LOGS"), level=int(os.getenv("LOG_LEVEL")), log_to_stdout=True
)

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/concepts")
def concepts(query: str = ""):
    df = plain_search.df

    if query:
        filtered = df[df["concept"].str.contains(query, case=False, na=False)]
    else:
        filtered = df

    items = filtered["concept"].dropna().astype(str).tolist()

    return {
        "total": int(len(items)),
        "items": items,
    }


@app.get("/search")
def search(query: str, semantic: bool = False, k: int = None):
    logger.info(f"Searching term: '{query}'")
    return plain_search.search(query, k=k)


@app.get("/predict")
def predict(concept: str, k: int = 200):
    logger.info(f"Predicting for concept: '{concept}'")
    return predictor.predict(concept, None, k)


@app.get(
    "/mentions",
    responses={
        503: {"description": "Mentions index not available on this backend instance"}
    },
)
def concept_mentions(concept: str, k: int = 10):
    logger.info(f"Loading mentions for concept: '{concept}'")

    if mentions is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Mentions index unavailable. Build it via 'python pimp_lookup.py' and "
                "set CONCEPT_WORK_IDS / WORKS_COMPACT if needed."
            ),
        )

    return mentions.get_mentions(concept=concept, k=k)
