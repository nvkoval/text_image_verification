# 🐾 Animal Text-Image Verification Pipeline

A multi-modal machine learning pipeline that combines **Named Entity Recognition (NER)** and **Image Classification** to verify if **animals mentioned in text are present in corresponding images**.
Example:
>  "There is a cow in the picture." + 🖼️ [image] → ✅ True / ❌ False

## Overview
This project implements a multi-modal ML pipeline that:
1. **Extracts animal names from text** using a fine-tuned BERT-based NER model.
2. **Classifies animals in images** using a fine-tuned EfficientNet model.
3. **Compares predictions** to determine if the mentioned animals are present in the image.

## 📁 Project Structure
```bash
animal_text_image_verification/
├── ner/                 # distilbert-NER model for extracting animal names
├── classifier/          # EfficientNet image classifier
├── pipeline/            # Combines NER + image model to verify truthfulness
├── notebooks/           # EDA and visualization
├── data/                # Dataset storage
├── models/              # Models checkpoints
├── requirements.txt     # Dependency list
└── README.md
```

## Features
- **Named Entity Recognition:** BERT-based model for extracting animal entities from natural language text
- **Image Classification:** EfficientNet-based model for classifying 10+ animal species
- **End-to-End Pipeline:** Automated workflow from text/image input to boolean output
- **Configurable Training:** Parameterized training scripts with YAML configuration
- **Comprehensive Evaluation:** Detailed analysis and visualization tools
- **Docker Support:** Containerized deployment for reproducibility


## Dataset
The project uses:
- **Image Dataset:** Adapted Animals-10 Dataset from Kaggle (10 animal classes)
- **NER Dataset:** Custom-generated dataset with animal entity annotations
- **Classes:** dog, cat, horse, spider, butterfly, chicken, sheep, cow, elephant, squirrel

## Model Architecture
### NER Model
- **Base Model:** distilbert-NER
- **Architecture:** Token classification head
- **Labels:** B-ANIMAL, O
- **Training:** Fine-tuned on custom animal entity dataset

### Image Classification Model
- **Base Model:** EfficientNet-B0
- **Architecture:** Classification head with 10 output classes
- **Input Size:** 224x224 RGB images
- **Training:** Fine-tuned with data augmentation
