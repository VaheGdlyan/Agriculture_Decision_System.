import logging
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths manager
# Assuming the script runs from the project root. We can also use Path(__file__).resolve().parents[1]
ROOT_DIR = Path('.') 
CONFIG_DIR = ROOT_DIR / 'configs'
YOLO_CONFIG_PATH = CONFIG_DIR / 'wheat_v8.yaml'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
YOLO_DIR = PROCESSED_DIR / 'yolo'

# shutil.make_archive automatically appends the '.zip' extension
ZIP_BASE_PATH = PROCESSED_DIR / 'yolo_dataset' 

YAML_CONTENT = """path: /content/dataset
train: images/train
val: images/val
names: {0: wheat_head}"""

def main():
    logger.info("Starting dataset export process...")
    
    # Step 1: Create YAML Config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Write the exact YAML structure required for YOLOv8 training
        with open(YOLO_CONFIG_PATH, 'w') as f:
            f.write(YAML_CONTENT)
        logger.info(f"Generated YOLO config at: {YOLO_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Failed to create YAML Config: {e}")
        sys.exit(1)
    
    # Step 2: Validation
    if not YOLO_DIR.exists() or not YOLO_DIR.is_dir():
        logger.error(f"Validation Error: The directory '{YOLO_DIR}' does not exist.")
        logger.error("Please run the prepare_yolo.py script first to generate the dataset!")
        sys.exit(1)
        
    logger.info(f"Verified dataset directory: {YOLO_DIR}")
    
    # Step 3: Compression
    logger.info(f"Compressing `{YOLO_DIR}` into a zip archive. This might take a moment...")
    try:
        # Make archive zips the contents of root_dir using format='zip'
        shutil.make_archive(
            base_name=str(ZIP_BASE_PATH),
            format='zip',
            root_dir=str(YOLO_DIR)
        )
        logger.info(f"Success! The dataset zip is ready for Cloud upload at: {ZIP_BASE_PATH}.zip")
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
