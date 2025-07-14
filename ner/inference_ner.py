import sys
from pathlib import Path
# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from ner.trainer_ner import AnimalNERTrainer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)


def extract_animals(text: str, model_path: str = './models/ner_model'):
    """
    Extract animal entities from text using NER model.

    Args:
        text: Input text to extract animals from
        model_path: Path to NER model

    Returns:
        List[str]: List of extracted animal names
    """
    try:
        logger.info("Initializing trainer...")
        trainer = AnimalNERTrainer()

        logger.info("Loading best model for prediction...")
        trainer.load_model(model_path)

        animals = trainer.extract_entities(text)
        return animals

    except Exception as e:
        logger.error(f"Error extracting animals: {e}")
        return []


# === USER PARAMETERS (edit these) ===
TEXT = "I saw a cat playing with a ball."
MODEL_PATH = "models/ner_model"

if __name__ == "__main__":
    try:
        animals = extract_animals(TEXT, MODEL_PATH)
        print("Extracted animal tokens:", animals)
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the NER model exists.")
