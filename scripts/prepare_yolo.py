import logging
import ast
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path('data/raw/gwhd2020')
CSV_PATH = Path('data/processed/train.csv')
OUTPUT_DIR = Path('data/processed/yolo')

def setup_directories():
    """Create YOLO directory structure (images/train, images/val, labels/train, labels/val)."""
    dirs = [
        OUTPUT_DIR / 'images' / 'train',
        OUTPUT_DIR / 'images' / 'val',
        OUTPUT_DIR / 'labels' / 'train',
        OUTPUT_DIR / 'labels' / 'val'
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")

def convert_coco_to_yolo(box, img_width=1024, img_height=1024):
    """
    Convert COCO bounding box [x_min, y_min, w, h] to 
    YOLO format [class_id, x_center, y_center, w_norm, h_norm].
    Class ID is assumed to be 0 for Wheat Heads.
    """
    x_min, y_min, w, h = box
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [0, x_center_norm, y_center_norm, w_norm, h_norm]

def main():
    logger.info("Starting YOLO dataset preparation...")
    setup_directories()
    
    if not CSV_PATH.exists():
        logger.error(f"CSV file not found at {CSV_PATH}. Please ensure the data exists.")
        return

    logger.info(f"Reading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Extract unique image ids
    unique_images = df['image_id'].unique()
    logger.info(f"Found {len(unique_images)} unique images.")
    
    # Train/Val split (80/20 strict reproducibility)
    train_ids, val_ids = train_test_split(unique_images, test_size=0.2, random_state=42)
    
    # Create a set for faster O(1) lookup
    train_ids_set = set(train_ids)
    
    # Image to source mapping (each image_id typically belongs to one source directory)
    img_to_source = df.groupby('image_id')['source'].first().to_dict()
    
    logger.info("Processing images and converting labels...")
    for img_id in tqdm(unique_images, desc="Preparing YOLO dataset"):
        split = 'train' if img_id in train_ids_set else 'val'
        
        # Get source folder for the current image
        source = img_to_source.get(img_id)
        if not source:
            logger.warning(f"No source found for image {img_id}, skipping.")
            continue
            
        # Check for both .jpg and .png extensions
        base_path = DATA_DIR / source / f"{img_id}"
        src_image_path = None
        ext = ""
        
        if base_path.with_suffix('.jpg').exists():
            src_image_path = base_path.with_suffix('.jpg')
            ext = ".jpg"
        elif base_path.with_suffix('.png').exists():
            src_image_path = base_path.with_suffix('.png')
            ext = ".png"
            
        if not src_image_path:
            logger.warning(f"Image not found (tried .jpg and .png) for ID {img_id} in {source}")
            continue
            
        dst_image_path = OUTPUT_DIR / 'images' / split / f"{img_id}{ext}"
        
        # Copy image
        shutil.copy2(src_image_path, dst_image_path)
            
        # Process labels for this image
        img_boxes = df[df['image_id'] == img_id]
        label_path = OUTPUT_DIR / 'labels' / split / f"{img_id}.txt"
        
        with open(label_path, 'w') as f:
            for _, row in img_boxes.iterrows():
                try:
                    # Support 'boxes' per instructions, but fallback to 'bbox' just in case
                    box_str = None
                    if 'boxes' in row:
                        box_str = row['boxes']
                    elif 'bbox' in row:
                        box_str = row['bbox']
                        
                    if pd.isna(box_str) or box_str is None:
                        continue
                        
                    # Parse string representation of list to actual Python list (list of lists)
                    boxes = ast.literal_eval(box_str)
                    
                    if not boxes or not isinstance(boxes, list):
                        continue
                        
                    # Handle both flat list and list of lists
                    if len(boxes) > 0 and isinstance(boxes[0], (int, float)):
                        boxes = [boxes]
                        
                    for box in boxes:
                        if not box or len(box) != 4:
                            continue
                            
                        yolo_box = convert_coco_to_yolo(box)
                        
                        # Format: class x_center y_center w h
                        line = f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}\n"
                        f.write(line)
                except Exception as e:
                    logger.error(f"Error processing box for image {img_id}: {e}")

    logger.info("YOLO dataset preparation completed successfully.")
    logger.info(f"YOLO structured data saved at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
