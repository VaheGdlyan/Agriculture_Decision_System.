from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/master_metadata.csv")
print(df.shape)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["source"]
)

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(out_dir / "train.csv", index=False)
test_df.to_csv(out_dir / "test.csv", index=False)

print("Dataset split is complete")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")