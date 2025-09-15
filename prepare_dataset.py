from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("go_emotions")

# Convert to DataFrame (train, validation, test)
for split in ["train", "validation", "test"]:
    df = pd.DataFrame(dataset[split])
    df.to_csv(f"goemotions_{split}.csv", index=False)

print("✅ Dataset downloaded and saved as CSVs.")
