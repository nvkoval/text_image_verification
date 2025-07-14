import torch
import logging
from ner.inference_ner import extract_animals
from classifier.inference_classifier import predict_image

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)

# Constants
CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog",
               "elephant", "horse", "sheep", "spider", "squirrel"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pipeline(text: str, image_path: str, img_model_path: str = '../models/image_classifier', ner_model_path: str = '../models/ner_model'):
    """
    Main pipeline function that combines NER and image classification.

    Args:
        text: Input text to extract animals from
        image_path: Path to the image file
        img_model_path: Path to image classifier model
        ner_model_path: Path to NER model

    Returns:
        bool: True if animals mentioned in text are present in image, False otherwise
    """
    try:
        # Extract animals from text
        animals_from_text = extract_animals(text, ner_model_path)

        if animals_from_text:
            logger.info(f"Animals found in text: {animals_from_text}")
        else:
            logger.info("No animals found in text.")
            return False

        # Classify image
        animal_from_image, confidence = predict_image(image_path, img_model_path)
        logger.info(f"Animal detected in image: {animal_from_image} (confidence: {confidence:.2f})")

        # Check if any animal from text is present in image
        result = any(animal.lower() in animal_from_image.lower() for animal in animals_from_text)
        return result

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        return False


def main():
    """Example usage of the pipeline."""
    # Example text and image
    text = "I see a sheep in the picture."
    image_path = '../data/images/test/sheep/OIP-jW9uZm9cox2kzddQLCN9NAHaE8.jpeg'

    logger.info(f"Input text: {text}")
    logger.info(f"Image path: {image_path}")

    # Run pipeline
    result = pipeline(text, image_path)

    # Output result
    logger.info(f'Pipeline result: {result}')


if __name__ == "__main__":
    main()
