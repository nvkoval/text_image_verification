import os
import gdown
import zipfile
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)

MODEL_DIR = 'data/models'
FOLDER_ID = '1zoRKKUBJkCzb0f4yZmPESH4vcHsmYgZL'

def load_model_if_needed(model_dir: str = MODEL_DIR,
                         folder_id: str = FOLDER_ID):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Models already exists at {model_dir}. Skipping download.")
        return
    logger.info("Downloading saved models from Google Drive...")
    gdown.download_folder(
        id=folder_id,
        output=model_dir,
        quiet=False,
        use_cookies=False
    )
    logger.info("Models ready.")

if __name__ == '__main__':
    load_model_if_needed()
