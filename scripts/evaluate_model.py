"""
Iteration 2 Evaluation Engine — Ground Truth vs. Prediction
=============================================================
Computes TP / FP / FN per image by comparing predicted boxes against YOLO
ground-truth labels at IoU >= 0.50.

Paths (all dynamic via pathlib):
  Weights  : outputs/iteration_2_tuned/iteration2_tuned.pt
  Images   : data/images/val/
  Labels   : data/labels/val/
  Outputs  : outputs/iteration_2_tuned/eval_metrics/
               ├── fn_coordinates.json
               ├── fp_coordinates.json
               └── error_metadata.csv
"""

import csv
import json
import logging
import sys
from pathlib import Path

import torch
from torchvision.ops import box_iou
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ CRITICAL: 'ultralytics' not found. Activate your venv first.")
    sys.exit(1)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Dynamic Paths ────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent

WEIGHTS      = ROOT / "outputs" / "iteration_2_tuned" / "eval_metrics" / "iteration_2_tuned.pt"
IMAGES_DIR   = ROOT / "data" / "processed" / "yolo" / "images" / "val"
LABELS_DIR   = ROOT / "data" / "processed" / "yolo" / "labels" / "val"
OUTPUT_DIR   = ROOT / "outputs" / "iteration_2_tuned" / "eval_metrics"

FN_JSON      = OUTPUT_DIR / "fn_coordinates.json"
FP_JSON      = OUTPUT_DIR / "fp_coordinates.json"
METADATA_CSV = OUTPUT_DIR / "error_metadata.csv"

IMG_SIZE       = 1024    # Normalisation reference for YOLO label denorm
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.50    # TP threshold

CSV_COLUMNS = [
    "Image_Name",
    "Total_Ground_Truth",
    "True_Positives",
    "False_Positives",
    "False_Negatives",
    "Avg_Confidence",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_yolo_label(label_path: Path) -> torch.Tensor:
    """Parse YOLO .txt → pixel xyxy tensor (N, 4). Returns (0,4) if empty."""
    try:
        lines = label_path.read_text().strip().splitlines()
    except Exception as e:
        logger.warning(f"   ⚠️  Could not read label {label_path.name}: {e}")
        return torch.zeros((0, 4), dtype=torch.float32)

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            xc, yc, w, h = [float(p) * IMG_SIZE for p in parts[1:5]]
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
        except ValueError:
            continue

    return (
        torch.tensor(boxes, dtype=torch.float32)
        if boxes
        else torch.zeros((0, 4), dtype=torch.float32)
    )


def find_label(stem: str) -> Path | None:
    p = LABELS_DIR / f"{stem}.txt"
    return p if p.exists() else None


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_evaluation():
    logger.info("=" * 62)
    logger.info("  📐 Iteration 2 Evaluation Engine — GT vs. Prediction")
    logger.info("=" * 62)

    # ── Infrastructure checks ───────────────────────────────────────────────────
    for name, path in [
        ("Weights",    WEIGHTS),
        ("Images dir", IMAGES_DIR),
        ("Labels dir", LABELS_DIR),
    ]:
        if not path.exists():
            logger.error(f"❌ {name} not found: {path}")
            sys.exit(1)

    val_images = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not val_images:
        logger.error(f"❌ No images found in {IMAGES_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Found {len(val_images)} validation images")

    # ── Load model ──────────────────────────────────────────────────────────────
    try:
        logger.info(f"🧠 Loading model: {WEIGHTS.name}")
        model = YOLO(str(WEIGHTS))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"   Running on: {device.upper()}")
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        sys.exit(1)

    # ── Accumulators ───────────────────────────────────────────────────────────
    csv_rows   = []
    fn_records = {}    # { img_path_str: [ [x1,y1,x2,y2], ... ] }
    fp_records = {}    # { img_path_str: [ { box, conf }, ... ] }
    total_tp_all = total_fp_all = total_fn_all = 0

    logger.info("🚀 Starting evaluation pass...")

    for img_path in tqdm(val_images, desc="Evaluating", unit="img"):
        stem = img_path.stem

        # ── Ground-Truth ────────────────────────────────────────────────────────
        label_path = find_label(stem)
        gt_boxes = (
            parse_yolo_label(label_path)
            if label_path
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        n_gt = gt_boxes.shape[0]

        # ── Inference ───────────────────────────────────────────────────────────
        try:
            with torch.no_grad():
                results = model.predict(
                    source=str(img_path), conf=CONF_THRESHOLD, verbose=False
                )
        except Exception as e:
            logger.warning(f"   ⚠️  Inference failed for {img_path.name}: {e}")
            continue

        pred_boxes  = results[0].boxes.xyxy.cpu()   # (M, 4)
        pred_scores = results[0].boxes.conf.cpu()   # (M,)
        n_pred = pred_boxes.shape[0]

        # ── IoU Matrix (N_gt × M_pred) ──────────────────────────────────────────
        if n_gt == 0 or n_pred == 0:
            iou_matrix = torch.zeros((max(n_gt, 1), max(n_pred, 1)))
        else:
            try:
                iou_matrix = box_iou(gt_boxes, pred_boxes)
            except Exception as e:
                logger.warning(f"   ⚠️  IoU failed for {img_path.name}: {e}")
                continue

        # ── FN (GT axis — dim=1 max) ─────────────────────────────────────────────
        image_fns = []
        if n_gt > 0:
            max_iou_per_gt = (
                iou_matrix.max(dim=1).values
                if n_pred > 0
                else torch.zeros(n_gt)
            )
            fn_mask  = max_iou_per_gt < IOU_THRESHOLD
            tp_count = int((~fn_mask).sum().item())
            fn_count = int(fn_mask.sum().item())
            image_fns = gt_boxes[fn_mask].tolist()
        else:
            tp_count = fn_count = 0

        # ── FP (Pred axis — dim=0 max) ───────────────────────────────────────────
        image_fps = []
        if n_pred > 0:
            max_iou_per_pred = (
                iou_matrix.max(dim=0).values
                if n_gt > 0
                else torch.zeros(n_pred)
            )
            fp_mask  = max_iou_per_pred < IOU_THRESHOLD
            fp_count = int(fp_mask.sum().item())
            for idx in range(n_pred):
                if fp_mask[idx]:
                    image_fps.append({
                        "box" : pred_boxes[idx].tolist(),
                        "conf": round(pred_scores[idx].item(), 4),
                    })
        else:
            fp_count = 0

        avg_conf = (
            round(float(pred_scores.mean().item()), 4) if n_pred > 0 else 0.0
        )

        if image_fns:
            fn_records[str(img_path)] = image_fns
        if image_fps:
            fp_records[str(img_path)] = image_fps

        total_tp_all += tp_count
        total_fp_all += fp_count
        total_fn_all += fn_count

        csv_rows.append({
            "Image_Name"        : img_path.name,
            "Total_Ground_Truth": n_gt,
            "True_Positives"    : tp_count,
            "False_Positives"   : fp_count,
            "False_Negatives"   : fn_count,
            "Avg_Confidence"    : avg_conf,
        })

    # ── Write outputs ───────────────────────────────────────────────────────────
    for path, data, label in [
        (FN_JSON, fn_records, "fn_coordinates.json"),
        (FP_JSON, fp_records, "fp_coordinates.json"),
    ]:
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"✅ {label} written ({len(data)} images)")
        except Exception as e:
            logger.error(f"❌ Failed to write {label}: {e}")

    try:
        with open(METADATA_CSV, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"✅ error_metadata.csv — {len(csv_rows)} rows")
    except Exception as e:
        logger.error(f"❌ Failed to write CSV: {e}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    denom_p = total_tp_all + total_fp_all
    denom_r = total_tp_all + total_fn_all
    precision = total_tp_all / denom_p if denom_p > 0 else 0.0
    recall    = total_tp_all / denom_r if denom_r > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    logger.info("=" * 62)
    logger.info("  📊 EVALUATION SUMMARY  (IoU threshold = 0.50)")
    logger.info("=" * 62)
    logger.info(f"   Total GT Boxes      : {(total_tp_all + total_fn_all):,}")
    logger.info(f"   True Positives (TP) : {total_tp_all:,}")
    logger.info(f"   False Positives (FP): {total_fp_all:,}")
    logger.info(f"   False Negatives (FN): {total_fn_all:,}")
    logger.info(f"   Precision @ IoU 0.5 : {precision:.4f}")
    logger.info(f"   Recall    @ IoU 0.5 : {recall:.4f}")
    logger.info(f"   F1 Score  @ IoU 0.5 : {f1:.4f}")
    logger.info(f"   Output dir          : {OUTPUT_DIR}")
    logger.info("=" * 62)
    logger.info("✅ Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
