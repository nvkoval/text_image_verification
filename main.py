"""
Main entry point for the Animal Text-Image Verification Pipeline.

This script demonstrates the complete pipeline that:
1. Extracts animal names from text using NER
2. Classifies animals in images using EfficientNet
3. Verifies if mentioned animals are present in the image
"""

import logging
from pipeline.pipeline import pipeline

# === USER PARAMETERS (edit these) ===
TEXT = "I see a sheep in the picture."
IMAGE_PATH = "data/images/test/sheep/OIP-jW9uZm9cox2kzddQLCN9NAHaE8.jpeg"
IMG_MODEL_PATH = "models/image_classifier"
NER_MODEL_PATH = "models/ner_model"
DEVICE = "auto"  # 'auto', 'cpu', or 'cuda'

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)


def main():
    # Set device if needed
    import constants
    constants.DEVICE = DEVICE

    logger.info("Starting Animal Text-Image Verification Pipeline...")
    logger.info(f"Text: {TEXT}")
    logger.info(f"Image: {IMAGE_PATH}")

    # Run pipeline
    result = pipeline(
        text=TEXT,
        image_path=IMAGE_PATH,
        img_model_path=IMG_MODEL_PATH,
        ner_model_path=NER_MODEL_PATH
    )

    # Output result
    logger.info(f'Pipeline result: {result}')
    print(result)


if __name__ == "__main__":
    main()
