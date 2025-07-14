import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import get_linear_schedule_with_warmup

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


class AnimalImageClassifier:
    """Trainer for Animal Image Classification."""
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = None,
        learning_rate: float = None,
        model_path: str = None,
        epochs: int = None
    ):
        # Use provided parameters or defaults from constants
        self.data_dir = Path(data_dir or constants.IMAGE_DATA_DIR)
        self.model_path = Path(model_path or constants.IMAGE_MODEL_PATH)
        self.batch_size = batch_size or constants.IMAGE_BATCH_SIZE
        self.learning_rate = learning_rate or constants.IMAGE_LEARNING_RATE
        self.epochs = epochs or constants.IMAGE_EPOCHS
        self.image_size = constants.IMAGE_SIZE
        self.num_classes = 10
        self.normalize_mean = constants.IMAGE_NORMALIZE_MEAN
        self.normalize_std = constants.IMAGE_NORMALIZE_STD

        # Ensure directories exist
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Get device from constants
        if constants.DEVICE == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(constants.DEVICE)

        logger.info(f"Using device: {self.device}")

        self._setup_transforms()
        self._setup_datasets()
        self._setup_model()

    def _setup_transforms(self):
        """Sets up the image transformations for training and validation."""
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean,
                                 std=self.normalize_std)
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean,
                                    std=self.normalize_std)
        ])

    def _setup_datasets(self):
        """Setup datasets using ImageFolder"""
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'

        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(f"Missing 'train' or 'val' directory in {self.data_dir}")

        logger.info(f"Loading datasets from {train_dir} and {val_dir}")

        self.train_dataset = ImageFolder(train_dir, transform=self.train_transforms)
        self.val_dataset = ImageFolder(val_dir, transform=self.val_transforms)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=4
        )

        self.class_names = self.train_dataset.classes
        self.num_classes = len(self.class_names)

        self.class_to_idx = self.train_dataset.class_to_idx
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        logger.info(f"Created Training DataLoader with {len(self.train_dataset)} samples.")
        logger.info(f"Created Validation DataLoader with {len(self.val_dataset)} samples.")

    def setup_test_loader(self, test_dir: str = None):
        """Setup test dataloader"""
        test_dir = Path(test_dir) if test_dir else self.data_dir / 'train'
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found in {self.test_dir}")

        self.test_dataset = ImageFolder(test_dir, transform=self.val_transforms)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=4
        )
        logger.info(f"Test loader initialized with {len(self.test_dataset)} samples.")

    def _setup_model(self):
        """Initializes the model, loss function, and optimizer."""
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features,
                                             self.num_classes)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        logger.info(f"Model initialized with {self.num_classes} classes.")

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> float:
        """Trains the model on the provided training data for one epoch."""
        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc='Training'):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluates the model on the validation set."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(eval_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return total_loss / len(eval_loader), {'accuracy': acc,
                                               'f1_score': f1}

    def evaluate_test(self):
        """Evaluates the model on the test set."""
        if not hasattr(self, 'test_loader'):
            raise AttributeError("Test loader not found. Call setup_test_loader() first")
        return self.evaluate(self.test_loader)

    def train(self, num_epochs: int = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.epochs

        best_f1 = 0.0
        train_losses, val_losses = [], []

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * num_epochs
        )

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_epoch(self.train_loader, optimizer, scheduler)
            val_loss, metrics = self.evaluate(self.val_loader)
            f1 = metrics['f1_score']

            logger.info(f"Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Score: {f1:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if f1 > best_f1:
                best_f1 = f1
                self.save_model(self.model_path)
                logger.info(f"Best model saved with F1 Score: {best_f1:.4f}")

        self.plot_training_history(train_losses, val_losses)
        logger.info("Training completed!")
        return train_losses, val_losses

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
    ):
        """Plots training and validation loss."""
        plt.figure(figsize=(7, 5))

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, loader=None):
        """Plots confusion matrix."""
        loader = loader or self.val_loader
        y_true, y_pred = [], []
        self.model.eval()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=-1).cpu().tolist()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(self.model_path / 'confusion_matrix.png')
        plt.show()

    def predict(self, image_path: str) -> Tuple[str, float]:
        """Predict class for a single image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transforms(image).unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        return self.idx_to_class[pred_idx], confidence

    def save_model(self, model_path: str):
        """Saves the model and tokenizer to the specified directory."""
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'idx_to_class': self.idx_to_class,
        }, Path(model_path) / 'model.pth')

    def load_model(self, model_path: str):
        """Loads saved model from the specified directory."""

        checkpoint = torch.load(Path(model_path) / 'model.pth',
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']
        self.idx_to_class = checkpoint['idx_to_class']
        logger.info(f"Model loaded from {model_path}")
