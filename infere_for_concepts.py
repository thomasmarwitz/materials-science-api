import logging
import sys
import os
from dotenv import load_dotenv
from predict.predict import Predictor
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

logger.info(
    f"Loading model from '{os.getenv('MODEL')}' with layers '{os.getenv('LAYERS')}'"
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


from pprint import pprint

concepts = sys.argv[1:]

for concept in concepts:
    results = predictor.predict(concept=concept, max_degree=None)

    print("Scores for concept", concept, "are:")
    pprint(results[:1000])
    print("\n\n\n")
