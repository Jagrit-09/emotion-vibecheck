import sqlite3
import pandas as pd
from datasets import Dataset

# Connect to your database
conn = sqlite3.connect("hinglish_goemotions.db")
cursor = conn.cursor()

# Read data from table
df = pd.read_sql_query("SELECT * FROM emotions", conn)

# Print few rows to verify
print(df.head())

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Save for later use
dataset.save_to_disk("hinglish_dataset")

print("✅ Dataset loaded and saved as Hugging Face Dataset format.")
