import torch
import sys
from pathlib import Path

# Allow running tests directly: `python tests/test_metrics.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import calculate_iou_matrix

def test_iou_logic():
    # Two identical boxes should have IoU of 1.0
    box1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    box2 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    
    iou = calculate_iou_matrix(box1, box2)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0)), f"Expected 1.0, got {iou[0,0]}"
    print("✅ IoU Logic Test: PASSED")

if __name__ == "__main__":
    test_iou_logic() 

