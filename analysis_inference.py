import logging
import sys
import os
from dotenv import load_dotenv
from predict.predict import Predictor
from ast import literal_eval
import gzip

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


def main(
    concepts="fixed_concepts.csv",
    output_dir="predictions",
):
    import pandas as pd
    from ast import literal_eval
    import pickle
    from pathlib import Path

    df = pd.read_csv(concepts)
    df["llama_concepts"] = df.llama_concepts.apply(literal_eval).apply(set)

    author_concepts = df.groupby("source")["llama_concepts"].agg(
        lambda sets: set().union(*sets)
    )

    df = author_concepts.reset_index()

    for source, concepts in zip(df.source, df.llama_concepts):
        logger.warning(f"Predicting for '{source}'")
        results = {}

        for concept in concepts:
            logger.info(f"Predicting for '{concept}'")
            try:
                results[concept] = predictor.predict(concept=concept, max_degree=None)
                logger.debug(f"Predicted for '{results[concept][:4]}'")
            except Exception as e:
                logger.error(f"Failed to predict for '{concept}'")
                logger.error(e)

        save_to = Path(output_dir) / f"{source}.pkl.gz"
        logger.info(f"Saving to '{save_to}'")

        with gzip.open(save_to, "wb") as f:
            pickle.dump(results, f)

        logger.warning(f"Predicted for '{source}'")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
