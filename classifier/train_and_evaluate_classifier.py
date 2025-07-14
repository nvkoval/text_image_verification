import sys
from pathlib import Path
# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from classifier.trainer_classifier import AnimalImageClassifier

# === USER PARAMETERS (edit these) ===
DATA_DIR = 'data/images'
MODEL_PATH = 'models/image_classifier'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = 'auto'

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)

def main():
    # Override constants with user parameters if provided
    if DEVICE:
        import constants
        constants.DEVICE = DEVICE

    logger.info("Starting image classifier training...")
    logger.info(f"User parameters: DATA_DIR={DATA_DIR}, MODEL_PATH={MODEL_PATH}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, EPOCHS={EPOCHS}, DEVICE={DEVICE}")

    # Initialize classifier with user parameters
    classifier = AnimalImageClassifier(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model_path=MODEL_PATH,
        epochs=EPOCHS
    )

    logger.info("Starting training...")
    classifier.train()

    logger.info("Evaluating on test set...")
    classifier.setup_test_loader()
    classifier.evaluate_test()
    classifier.plot_confusion_matrix(classifier.test_loader)


if __name__ == "__main__":
    main()
