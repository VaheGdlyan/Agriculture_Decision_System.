import ast
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class WheatDataset(Dataset):
    """
    Dataset for GWHD metadata CSV with columns:
    image_id, path, width, height, bbox_count, boxes
    where boxes are COCO xywh: [x, y, w, h].
    """

    def __init__(self, csv_file, root_dir=".", transforms=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Support both path styles:
        # 1) CSV already contains project-relative/full path (e.g. data/raw/gwhd2020/...)
        # 2) CSV contains relative path under root_dir (e.g. arvalis_1/....png)
        raw_path = Path(str(row["path"]))
        img_path = raw_path if raw_path.exists() else (self.root_dir / raw_path)

        if not img_path.exists():
            raise FileNotFoundError(f"Could not find image at: {img_path.resolve()}")

        image = Image.open(img_path).convert("RGB")

        # Parse boxes from CSV string -> list[[x, y, w, h], ...]
        boxes_xywh = ast.literal_eval(row["boxes"]) if isinstance(row["boxes"], str) else row["boxes"]
        boxes_xyxy = []
        
        for box in boxes_xywh:
            x, y, w, h = [float(v) for v in box]
            # Convert COCO [x, y, w, h] to Pascal VOC [x1, y1, x2, y2] for PyTorch
            boxes_xyxy.append([x, y, x + w, y + h])

        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # class 1: wheat_head
        
        # Calculate area for the COCO evaluator
        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
            
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64), # Using index as unique ID
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            try:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes_xyxy,
                    labels=labels.tolist(),
                )
                image = transformed["image"]
                target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)
            except Exception:
                # Fallback for basic torchvision transforms
                image = self.transforms(image)
        else:
            image = self.to_tensor(image)

        return image, target

def detection_collate_fn(batch):
    return tuple(zip(*batch))

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Correct paths based on your Cursor sidebar
    CSV_PATH = "data/processed/train.csv"
    IMAGE_DIR = "."

    print(f"📂 Checking Dataset logic...")
    
    try:
        dataset = WheatDataset(csv_file=CSV_PATH, root_dir=IMAGE_DIR)
        
        if len(dataset) == 0:
            print("⚠️ CSV is empty! Check data/processed/train.csv")
        else:
            img, tar = dataset[0]
            print("\n" + "="*30)
            print("✅ SUCCESS: Dataset Loaded!")
            print(f"📊 Total Images: {len(dataset)}")
            print(f"🖼️  Image Shape: {img.shape}")
            print(f"🌾 Boxes in first image: {len(tar['boxes'])}")
            print("="*30 + "\n")
            
    except FileNotFoundError as e:
        print(f"\n❌ PATH ERROR: {e}")
    except Exception as e:
        print(f"\n❌ LOGIC ERROR: {e}")