import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from src.dataset import WheatDataset
from src.model import get_wheat_model
from src.engine import train_one_epoch

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # 0. Infrastructure
    DEVICE = torch.device('cpu') 
    ROOT_DIR = Path("data/raw/gwhd2020")
    CSV_PATH = Path("data/processed/train.csv")
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load Data
    print("📂 Loading Data...")
    df = pd.read_csv(CSV_PATH)
    full_dataset = WheatDataset(df, root_dir=ROOT_DIR)
    
    # SPEED HACK for baseline verification
    indices = torch.arange(20)
    train_subset = Subset(full_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 2. Model & Optimizer
    print("🧠 Initializing Faster R-CNN Baseline...")
    model = get_wheat_model(num_classes=2).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 3. Train
    print("🚀 Starting Baseline Local Run (3 Epochs)...")
    for epoch in range(3):
        avg_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
    # 4. Save
    save_path = "outputs/baseline_quick.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Baseline Finished. Weights saved to {save_path}")

if __name__ == "__main__":
    main()