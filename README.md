# Animal Text-Image Verification Pipeline

A multi-modal machine learning pipeline that combines **Named Entity Recognition (NER)** and **Image Classification** to verify if **animals mentioned in text are present in corresponding images**.

**Example:**
> "There is a cow in the picture." + 🖼️ [image] → ✅ True / ❌ False

## Overview
This project implements a multi-modal ML pipeline that:
1. **Extracts animal names from text** using a fine-tuned BERT-based NER model
2. **Classifies animals in images** using a fine-tuned EfficientNet model
3. **Compares predictions** to determine if the mentioned animals are present in the image

## 📁 Project Structure
```
text_image_verification/
├── classifier/                 # Image classification module
│   ├── trainer_classifier.py   # Training script for image classifier
│   ├── train_and_evaluate_classifier.py  # Main training script
│   ├── inference_classifier.py # Inference script
│   └── download_image.py       # Image dataset download script
├── ner/                        # Named Entity Recognition module
│   ├── trainer_ner.py          # Training script for NER model
│   ├── train_and_evaluate_ner.py  # Main training script
│   └── inference_ner.py        # Inference script
├── pipeline/                   # Main pipeline
│   └── pipeline.py             # Combined text-image verification
├── models/                     # Trained models
│   ├── image_classifier/       # Image classification model
│   └── ner_model/              # NER model
├── data/                       # Dataset storage
├── constants.py                # Default configuration values
├── main.py                     # Main pipeline script
└── requirements.txt            # Python dependencies
```

## Features

- **Text Processing**: Named Entity Recognition to extract animal names from text
- **Image Classification**: EfficientNet-based model to classify animals in images
- **Verification Pipeline**: Combines both models to verify text-image consistency
- **Simple Configuration**: All parameters are set by editing variables at the top of each script

## Dataset
The project uses:
- **Image Dataset:** Adapted Animals-10 Dataset from Kaggle (10 animal classes)
- **NER Dataset:** Custom-generated dataset with animal entity annotations
- **Classes:** dog, cat, horse, spider, butterfly, chicken, sheep, cow, elephant, squirrel

### Image Dataset
Due to GitHub file size limits, the image dataset is **stored externally on Google Drive**.
Download it with:
```bash
pip install gdown
python classifier/download_image.py
```

Or manually from [Google Drive](https://drive.google.com/file/d/10bGs8aTsRttHz7K6T5KP-FyAe-nxOynb/) and unzip it into `data/images/`.

Organize your image dataset in the following structure:
```
data/images/
├── train/
│   ├── cat/
│   ├── dog/
│   └── ...
├── val/
│   ├── cat/
│   ├── dog/
│   └── ...
└── test/
    ├── cat/
    ├── dog/
    └── ...
```

- The script `classifier/download_image.py` will download and extract the dataset to `data/images/` automatically. If the dataset already exists, it will skip the download.
- The zip file is temporarily stored in `data/test/animals.zip` and deleted after extraction.

### NER Dataset
Provide JSON files with the following format:
```json
[
  {
    "tokens": ["I", "saw", "a", "cat", "playing"],
    "labels": ["O", "O", "O", "B-ANIMAL", "O"]
  }
]
```

## Model Architecture

- **Image Classifier**: EfficientNet-B0 with pretrained weights
- **NER Model**: DistilBERT-NER with token classification head
- **Pipeline**: Combines both models for verification

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text_image_verification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Downloading the Image Dataset

Run the following to download and extract the image dataset:
```bash
python classifier/download_image.py
```
- The script will skip downloading if the dataset is already present in `data/images/`.
- The zip file is deleted after extraction.

### 2. Training Image Classifier

Edit the variables at the top of `classifier/train_and_evaluate_classifier.py` to set your data directory, model path, batch size, learning rate, epochs, and device.

Then run:
```bash
python classifier/train_and_evaluate_classifier.py
```

### 3. Training NER Model

Edit the variables at the top of `ner/train_and_evaluate_ner.py` to set your data directory, model path, model name, batch size, learning rate, epochs, max length, and device.

Then run:
```bash
python ner/train_and_evaluate_ner.py
```

### 4. Running the Pipeline

Edit the variables at the top of `main.py` to set your input text, image path, model paths, and device. Example:
```python
# === USER PARAMETERS (edit these) ===
TEXT = "I see a sheep in the picture."
IMAGE_PATH = "data/images/test/sheep/OIP-jW9uZm9cox2kzddQLCN9NAHaE8.jpeg"
IMG_MODEL_PATH = "models/image_classifier"
NER_MODEL_PATH = "models/ner_model"
DEVICE = "auto"
```
Then run:
```bash
python main.py
```
The result (True/False) will be printed to the console.

### 5. Individual Model Inference

#### Image Classification
Edit the variables at the top of `classifier/inference_classifier.py`:
```python
IMAGE_PATH = "data/images/test/sheep/OIP-jW9uZm9cox2kzddQLCN9NAHaE8.jpeg"
MODEL_PATH = "./models/image_classifier"
SHOW_IMAGE = False
```
Then run:
```bash
python classifier/inference_classifier.py
```
- Prints: `Predicted class: <class>, Confidence: <confidence>`

#### NER Extraction
Edit the variables at the top of `ner/inference_ner.py`:
```python
TEXT = "I saw a cat playing with a ball."
MODEL_PATH = "models/ner_model"
```
Then run:
```bash
python ner/inference_ner.py
```
- Prints: `Extracted animal tokens: ['cat']`

## Configuration

All default parameters are defined in `constants.py`. You can modify these defaults, but the main way to set parameters is to edit the variables at the top of each script (see above).

## Notes

- Ensure all required variables are set before running scripts.
- Use correct relative or absolute paths for all data and model files.
- For Jupyter notebooks, add the project root to `sys.path` for imports if needed.
- See the EDA notebook for exploratory data analysis and further customization.
