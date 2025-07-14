import sys
from pathlib import Path
# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import json
import logging
import sys
from pathlib import Path
from ner.trainer_ner import AnimalNERTrainer



# === USER PARAMETERS (edit these) ===
DATA_DIR = 'data/texts'
MODEL_PATH = '../models/ner_model'
MODEL_NAME = 'dslim/distilbert-NER'
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 3
MAX_LENGTH = 182
DEVICE = 'auto'

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Override constants with user parameters if provided
    if DEVICE:
        import constants
        constants.DEVICE = DEVICE

    logger.info("Starting NER training...")
    logger.info(f"User parameters: DATA_DIR={DATA_DIR}, MODEL_PATH={MODEL_PATH}, MODEL_NAME={MODEL_NAME}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, EPOCHS={EPOCHS}, MAX_LENGTH={MAX_LENGTH}, DEVICE={DEVICE}")

    # Initialize trainer with user parameters
    trainer = AnimalNERTrainer(
        model_name=MODEL_NAME,
        data_dir=DATA_DIR,
        model_path=MODEL_PATH,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        epochs=EPOCHS
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Loading best model for evaluation...")
    trainer.load_model()

    logger.info("Evaluating on test set...")
    test_data = load_json(os.path.join(trainer.data_dir, 'test_ner.json'))
    test_loader = trainer.create_dataloader(test_data,
                                            batch_size=trainer.batch_size,
                                            shuffle=False)
    test_loss, metrics = trainer.evaluate(test_loader)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Test precision: {metrics['precision']:.4f}")
    logger.info(f"Test recall: {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
