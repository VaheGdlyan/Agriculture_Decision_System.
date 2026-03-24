import ast
import pandas as pd
import torch
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class WheatDataset(Dataset):
    """
    Dataset for GWHD metadata CSV with columns:
    image_id, path, width, height, bbox_count, boxes
    """

    def __init__(self, csv_file, root_dir=".", transforms=None):
        # FIX: Robustly handle both a path string and an already loaded DataFrame
        if isinstance(csv_file, (str, Path)):
            self.df = pd.read_csv(str(csv_file))
        else:
            self.df = csv_file
            
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # FIX: Ensure idx is a plain integer (converts torch.Tensor to int)
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        row = self.df.iloc[idx]

        # FIX: Ensure we are joining the path correctly
        # If row["path"] is 'arvalis_1/ethz_1.jpg', it joins with root_dir
        raw_path_str = str(row["path"])
        img_path = self.root_dir / raw_path_str

        if not img_path.exists():
            # Fallback check: maybe the path in CSV is already absolute?
            if Path(raw_path_str).exists():
                img_path = Path(raw_path_str)
            else:
                raise FileNotFoundError(f"Could not find image at: {img_path.resolve()}")

        image = Image.open(img_path).convert("RGB")

        # Parse boxes from CSV string -> list[[x, y, w, h], ...]
        boxes_xywh = ast.literal_eval(row["boxes"]) if isinstance(row["boxes"], str) else row["boxes"]
        boxes_xyxy = []
        
        for box in boxes_xywh:
            x, y, w, h = [float(v) for v in box]
            # Convert COCO [x, y, w, h] to Pascal VOC [x1, y1, x2, y2]
            boxes_xyxy.append([x, y, x + w, y + h])

        # Handle images with zero boxes to avoid training crashes
        if len(boxes_xyxy) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
            
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # class 1: wheat_head
        
        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
            
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        # Handle Transformations
        if self.transforms:
            try:
                # Assuming Albumentations format
                sample = self.transforms(image=np.array(image), bboxes=boxes.tolist(), labels=labels.tolist())
                image = sample['image']
                target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
            except:
                image = self.transforms(image)
        else:
            image = self.to_tensor(image)

        return image, target

def detection_collate_fn(batch):
    return tuple(zip(*batch))