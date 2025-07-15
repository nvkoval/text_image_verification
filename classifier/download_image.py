import os
import gdown
import zipfile
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)

ZIP_PATH = 'data/test/animals.zip'
EXTRACT_DIR = 'data/test/images'
FILE_ID = '10bGs8aTsRttHz7K6T5KP-FyAe-nxOynb'  # ← Replace with your actual file ID

def load_dataset_if_needed(
    file_id: str = FILE_ID,
    zip_path: str = ZIP_PATH,
    extract_dir: str = EXTRACT_DIR):
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        logger.info(f"Dataset already exists at {extract_dir}. Skipping download.")
        return

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"

    logger.info("Downloading ZIP file from Google Drive...")
    gdown.download(url, output=zip_path, quiet=False)

    logger.info("Extracting ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    logger.info("Removing ZIP file...")
    os.remove(zip_path)

    logger.info("Dataset ready.")

if __name__ == '__main__':
    load_dataset_if_needed()
