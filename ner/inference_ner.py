from trainer_ner import AnimalNERTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "./models/ner_model"


def extract_animals(text):
    logger.info("Initializing trainer...")
    trainer = AnimalNERTrainer()

    logger.info("Loading best model for prediction...")
    trainer.load_model(MODEL_DIR)

    animals = trainer.extract_entities(text)

    return animals


if __name__ == "__main__":
    sample_text = "I saw a cat playing with a ball."
    print("Extracted animal tokens:", extract_animals(sample_text))
