from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search.search import SemanticSearch, PlainSearch
from predict.predict import Predictor
from predict.generation import Generator, OpenAi
from predict.graph import Graph
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

sem_search = SemanticSearch(
    logger=logger,
    embeddings=os.getenv("SEMANTIC_EMBEDDINGS"),
    model_name=os.getenv("NLP_MODEL"),
)
plain_search = PlainSearch(
    logger=logger,
    lookup=os.getenv("LOOKUP"),
)


openai = OpenAi(
    logger=logger,
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG"),
)

logger.info(f"Loading graph from '{os.getenv('GRAPH')}'")
G = Graph.from_path(os.getenv("GRAPH"))

predictor = Predictor(
    logger=logger,
    lookup=os.getenv("LOOKUP"),
    feature_embeddings=os.getenv("FEATURE_EMBEDDINGS"),
    concept_embeddings=os.getenv("CONCEPT_EMBEDDINGS"),
    graph=G,
    since=int(os.getenv("SINCE")),
    layers=os.getenv("LAYERS"),
    model=os.getenv("MODEL"),
)
generator = Generator(
    logger=logger,
    graph=G,
    since=int(os.getenv("SINCE")),
    lookup_file="data/lookup/lookup.M.new.csv",
    prompt_file="data/prompt.txt",
    api=openai,
)

logger.debug("Freeing graph from memory")
del G


@app.get("/search")
def search(query: str, semantic: bool = False, k: int = None):
    logger.info(f"Searching term: '{query}'")
    if semantic:
        return sem_search.search(query, k=k)
    else:
        return plain_search.search(query, k=k)


@app.get("/predict")
async def predict(
    concept: str, max_degree: int = None, min_depth: int = None, k: int = 10
):
    logger.info(f"Predicting for concept: '{concept}'")
    return await predictor.predict(concept, max_degree, min_depth, k)


# @app.get("/predict_pair")
# def predict_pair(concept_a: str, concepts_b: str):
#     return None


@app.get("/generate_abstracts")
def generate_abstracts(
    concept_a: str, concept_b: str, k: int = 3, min_words=100, max_words=150
):
    if k > 10:  # not allowed
        k = 10

    if max_words > 300:  # not allowed
        max_words = 300

    response = generator.generate_abstracts(
        concept_a, concept_b, k=k, min_words=min_words, max_words=max_words
    )
    return response
