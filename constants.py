# Default configuration values for the text-image verification pipeline

# General settings
RANDOM_SEED = 42
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEVICE = "auto"  # "auto", "cpu", or "cuda"

# Image Classifier settings
IMAGE_DATA_DIR = "data/images"
IMAGE_MODEL_PATH = "models/image_classifier"
IMAGE_BATCH_SIZE = 32
IMAGE_LEARNING_RATE = 2e-4
IMAGE_SIZE = 224
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]
IMAGE_EPOCHS = 10

# NER Model settings
NER_MODEL_NAME = "dslim/distilbert-NER"
NER_DATA_DIR = "data/texts"
NER_MODEL_PATH = "models/ner_model"
NER_BATCH_SIZE = 16
NER_LEARNING_RATE = 5e-5
NER_MAX_LENGTH = 128
NER_EPOCHS = 3
ANIMAL_CLASSES = ["cat", "dog", "horse", "cow", "sheep", "elephant", "butterfly", "chicken", "spider", "squirrel"]
