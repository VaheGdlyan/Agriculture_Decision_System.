import os
import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

BASE_PATH = "data/raw/gwhd2020"
OUTPUT_CSV = "data/raw/master_metadata.csv"

all_records = []

# Subfolders: arvalis_1, ethz_1, ... each has a matching {name}.json (COCO format)
subsets = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]

print(f"🚀 Processing {len(subsets)} sub-datasets...")

for source in tqdm(subsets):
    json_path = os.path.join(BASE_PATH, f"{source}.json")
    img_folder = os.path.join(BASE_PATH, source)

    if not os.path.exists(json_path):
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # COCO: images[], annotations[] with image_id -> bbox [x, y, w, h]
    if not isinstance(coco, dict) or "images" not in coco or "annotations" not in coco:
        print(f"⚠️  Skipping {source}: not COCO-style JSON (expected keys: images, annotations)")
        continue

    boxes_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        boxes_by_image[ann["image_id"]].append(ann["bbox"])

    for img in coco["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        boxes = boxes_by_image.get(img_id, [])

        rel_path = os.path.join(BASE_PATH, source, file_name)

        all_records.append(
            {
                "image_id": file_name.replace(".png", ""),
                "coco_image_id": img_id,
                "source": source,
                "path": rel_path,
                "width": img.get("width"),
                "height": img.get("height"),
                "bbox_count": len(boxes),
                "boxes": json.dumps(boxes),  # list of [x, y, w, h] in pixels
            }
        )

df = pd.DataFrame(all_records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Done! Found {len(df)} total images.")
print(f"📍 File saved at: {OUTPUT_CSV}")
