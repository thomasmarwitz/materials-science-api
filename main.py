from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search.search import SemanticSearch, PlainSearch
from predict.predict import Predictor
import logging
import sys
import os
from dotenv import load_dotenv

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


logger = setup_logger(os.getenv("LOGS"), level=logging.INFO, log_to_stdout=True)

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sem_search = SemanticSearch(
    logger=logger,
    embeddings=os.getenv("SEMANTIC_EMBEDDINGS"),
    model_name=os.getenv("NLP_MODEL"),
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
    graph=os.getenv("GRAPH"),
    since=int(os.getenv("SINCE")),
    layers=os.getenv("LAYERS"),
    model=os.getenv("MODEL"),
)


@app.get("/search")
def search(query: str, semantic: bool = False, k: int = 10):
    logger.info(f"Searching term: '{query}'")
    if semantic:
        return sem_search.search(query, k=k)
    else:
        return plain_search.search(query, k=k)


@app.get("/predict")
def predict(concept: str, max_degree: int = None, max_depth: int = None, k: int = 10):
    return None


@app.get("/predict_pair")
def predict_pair(concept_a: str, concepts_b: str):
    return None
