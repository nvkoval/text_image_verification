import torch
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

torch.manual_seed(7)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_model(model_path: str):
    """Load image classification model from checkpoint."""
    try:
        model = efficientnet_b0()
        checkpoint = torch.load(Path(model_path) / 'model.pth',
                                map_location=DEVICE)
        num_classes = checkpoint['num_classes']
        idx_to_class = checkpoint['idx_to_class']
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                        num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        logger.info(f"Model loaded from {model_path}")
        return model, idx_to_class
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def predict_image(image_path: str, model_path: str = './models/image_classifier', show_image: bool = False):
    """
    Predict animal class in image.
    Args:
        image_path: Path to the image file
        model_path: Path to model checkpoint
        show_image: Whether to display the image (default: False)
    Returns:
        tuple: (predicted_class, confidence)
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        if show_image:
            plt.imshow(image)
            plt.axis("off")
            plt.title("Input Image")
            plt.show()

        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Load model and predict
        model, idx_to_class = load_image_model(model_path)
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())

        return idx_to_class[pred_idx], confidence

    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        raise


# === USER PARAMETERS (edit these) ===
IMAGE_PATH = "data/images/test/sheep/OIP-jW9uZm9cox2kzddQLCN9NAHaE8.jpeg"
MODEL_PATH = "./models/image_classifier"
SHOW_IMAGE = False

if __name__ == "__main__":
    try:
        img_class, confidence = predict_image(IMAGE_PATH, MODEL_PATH, SHOW_IMAGE)
        print(f"Predicted class: {img_class}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the model and test image exist.")
