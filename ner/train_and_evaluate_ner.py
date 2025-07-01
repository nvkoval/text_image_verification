import os
import json
from trainer_ner import AnimalNERTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "./data/texts"
MODEL_DIR = "./models/ner_model"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    logger.info("Loading datasets...")
    train_data = load_json(os.path.join(DATA_DIR, 'train_ner.json'))
    val_data = load_json(os.path.join(DATA_DIR, 'val_ner.json'))
    test_data = load_json(os.path.join(DATA_DIR, 'test_ner.json'))

    logger.info("Initializing trainer...")
    trainer = AnimalNERTrainer()

    logger.info("Starting training...")
    trainer.train(train_data,
                  val_data,
                  save_path=MODEL_DIR,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE)

    logger.info("Loading best model for evaluation...")
    trainer.load_model(MODEL_DIR)

    logger.info("Evaluating on test set...")
    test_loader = trainer.create_dataloader(test_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
    test_loss, metrics = trainer.evaluate(test_loader)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")

    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
