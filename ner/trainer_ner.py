import os
import re
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
import constants

# Set random seed from constants
torch.manual_seed(constants.RANDOM_SEED)

# Setup logging
logging.basicConfig(level=getattr(logging, constants.LOG_LEVEL),
                    format=constants.LOG_FORMAT,
                    force=True)
logger = logging.getLogger(__name__)


class AnimalNERDataset(Dataset):
    """Dataset class for Animal Named Entity Recognition (NER) task."""

    def __init__(
            self,
            data: List[Dict],
            tokenizer,
            label_to_id: Dict[str, int],
            max_length: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            is_split_into_words=True
        )

        aligned_labels = self.align_labels_with_tokens(labels, encoding)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

    def align_labels_with_tokens(
        self,
        original_labels: List[str],
        encoding
    ) -> List[int]:
        """Aligns NER labels with tokenized input."""
        aligned_labels = []
        word_ids = encoding.word_ids(batch_index=0)

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx < len(original_labels):
                aligned_labels.append(
                    self.label_to_id.get(original_labels[word_idx], 'O')
                )
            else:
                aligned_labels.append(-100)

        return aligned_labels


class AnimalNERTrainer:
    """Main trainer class for Animal Ner model."""

    def __init__(
        self,
        model_name: str = None,
        animal_classes: List[str] = None,
        data_dir: str = None,
        model_path: str = None,
        batch_size: int = None,
        learning_rate: float = None,
        max_length: int = None,
        epochs: int = None
    ):
        # Use provided parameters or defaults from constants
        self.model_name = model_name or constants.NER_MODEL_NAME
        self.animal_classes = animal_classes or constants.ANIMAL_CLASSES
        self.data_dir = data_dir or constants.NER_DATA_DIR
        self.model_path = model_path or constants.NER_MODEL_PATH
        self.batch_size = batch_size or constants.NER_BATCH_SIZE
        self.learning_rate = learning_rate or constants.NER_LEARNING_RATE
        self.max_length = max_length or constants.NER_MAX_LENGTH
        self.epochs = epochs or constants.NER_EPOCHS

        self.label_to_id = {'O': 0, 'B-ANIMAL': 1}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # Get device from constants
        if constants.DEVICE == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(constants.DEVICE)

        logger.info(f"Using device: {self.device}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_to_id),
            ignore_mismatched_sizes=True
        ).to(self.device)

    def load_data(self, file_path: str) -> List[Dict]:
        """Loads the dataset from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data

    def create_dataloader(
        self,
        data: List[Dict],
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Creates a DataLoader from data."""
        dataset = AnimalNERDataset(
            data,
            tokenizer=self.tokenizer,
            label_to_id=self.label_to_id
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> float:
        """Trains the model on the provided training data for one epoch."""
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(
        self,
        eval_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluates the model on the validation set."""
        self.model.eval()
        all_predictions, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

                predictions = torch.argmax(outputs.logits,
                                           dim=-1).cpu().numpy()
                labels = labels.cpu().numpy()

                # Remove ignored labels (-100)
                mask = labels != -100

                all_predictions.extend(predictions[mask])
                all_labels.extend(labels[mask])

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        return total_loss / len(eval_loader), {'f1_score': f1,
                                               'precision': precision,
                                               'recall': recall}

    def train(
        self,
        train_data: List[Dict] = None,
        val_data: List[Dict] = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        save_path: str = None
    ):
        """Main training loop."""
        # Use provided parameters or defaults
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate
        save_path = save_path or self.model_path

        # Load data if not provided
        if train_data is None:
            train_data = self.load_data(os.path.join(self.data_dir, 'train_ner.json'))
        if val_data is None:
            val_data = self.load_data(os.path.join(self.data_dir, 'val_ner.json'))

        train_loader = self.create_dataloader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True)
        val_loader = self.create_dataloader(val_data,
                                            batch_size=batch_size,
                                            shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )

        best_f1 = 0.0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Train the model for one epoch
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"Training Loss: {train_loss:.4f}")

            val_loss, metrics = self.evaluate(val_loader)
            f1 = metrics['f1_score']
            logger.info(f"Validation Loss: {val_loss:.4f}, F1 Score: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                self.save_model(save_path)
                logger.info(
                    f"Save best model to {save_path} with F1: {best_f1:.4f}."
                )

        logger.info("Training completed!")

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a sentence into words, removing punctuation
            and converting to lowercase.
        """
        return re.findall(r'\w+', text.lower())

    def predict(self, text: str) -> List[Tuple[str, str]]:
        """Predicts NER labels for a given text."""
        self.model.eval()
        text = self.tokenize(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
            is_split_into_words=True
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits,
                                       dim=-1)[0].cpu().tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_labels = [
            self.id_to_label[pred] for pred in predictions
        ]

        result = [
            (token, label)
            for token, label in zip(tokens, predicted_labels)
            if token not in ['[CLS]', '[SEP]', '[PAD]']
        ]
        return result

    def extract_entities(self, text: str) -> List[str]:
        """Extracts animal entities from text."""
        predictions = self.predict(text)

        entities = [
            token.replace('##', '') for token, label in predictions
            if label == 'B-ANIMAL'
        ]
        return entities

    def save_model(self, save_path: str = None):
        """Saves the model and tokenizer to the specified directory."""
        if save_path is None:
            save_path = self.model_path

        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        with open(os.path.join(save_path, 'label_mappings.json'), 'w') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }, f, indent=4)

    def load_model(self, model_path: str = None):
        """Loads saved model and tokenizer from the specified directory."""
        if model_path is None:
            model_path = self.model_path

        self.model = DistilBertForTokenClassification.from_pretrained(
            model_path
        ).to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

        with open(os.path.join(model_path, 'label_mappings.json'), 'r') as f:
            mappings = json.load(f)
        self.label_to_id = mappings['label_to_id']
        self.id_to_label = {
            int(k): v for k, v in mappings['id_to_label'].items()
            }

        logger.info(f"Model loaded from {model_path}")
