import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ CRITICAL: 'ultralytics' library not found. Is your venv active?")
    sys.exit(1)

# --- 1. DYNAMIC INFRASTRUCTURE PATHING ---
ROOT_DIR = Path(__file__).resolve().parent.parent

BASELINE_WEIGHTS = ROOT_DIR / 'outputs' / 'yolov8_baseline_640' / 'best.pt'
TUNED_WEIGHTS = ROOT_DIR / 'outputs' / 'iteration_2_tuned' / 'iteration_2_tuned.pt'
TEST_DIR = ROOT_DIR / 'data' / 'test_samples'

# We will create dedicated gallery folders for a clean comparison
BASELINE_OUT_DIR = ROOT_DIR / 'outputs' / 'iteration_2_tuned' / 'eval_metrics' / 'baseline_gallery'
TUNED_OUT_DIR = ROOT_DIR / 'outputs' / 'iteration_2_tuned' / 'eval_metrics' / 'tuned_gallery'

CONFIDENCE_THRESHOLD = 0.25

def run_batch_audit():
    print("\n" + "="*60)
    print("🚀 IGNITING MULTI-IMAGE BATCH SHOWDOWN")
    print("="*60)
    
    # Grab all images in the folder (jpg, jpeg, png)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    test_images = [f for f in TEST_DIR.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not test_images:
        print(f"❌ Error: No images found in {TEST_DIR}. Please add some test images!")
        sys.exit(1)

    print(f"📦 Found {len(test_images)} test images. Loading engines...\n")

    try:
        baseline_model = YOLO(str(BASELINE_WEIGHTS))
        tuned_model = YOLO(str(TUNED_WEIGHTS))
    except Exception as e:
        print(f"❌ FATAL ERROR loading YOLO architectures: {e}")
        sys.exit(1)

    # --- EXECUTE LOOP ---
    for img_path in test_images:
        print(f"📸 Processing: {img_path.name}")
        
        # 1. Baseline Predict
        baseline_model.predict(
            source=str(img_path), 
            conf=CONFIDENCE_THRESHOLD, 
            save=True, 
            project=str(BASELINE_OUT_DIR), 
            name='preds', # Saves inside baseline_gallery/preds
            exist_ok=True 
        )
        
        # 2. Tuned Predict
        tuned_model.predict(
            source=str(img_path), 
            conf=CONFIDENCE_THRESHOLD, 
            save=True, 
            project=str(TUNED_OUT_DIR), 
            name='preds', # Saves inside tuned_gallery/preds
            exist_ok=True 
        )

    print("\n" + "="*60)
    print("🏆 BATCH AUDIT COMPLETE!")
    print(f"Baseline Gallery: {BASELINE_OUT_DIR / 'preds'}")
    print(f"Tuned Gallery: {TUNED_OUT_DIR / 'preds'}")
    print("Open them side-by-side to see the architectural differences.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_batch_audit()