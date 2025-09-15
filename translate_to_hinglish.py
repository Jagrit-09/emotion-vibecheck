from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm

# Load model and tokenizer
model_name = "ai4bharat/indictrans2-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the train dataset
df = pd.read_csv("goemotions_train.csv")

# Translate function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(**inputs, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Translate and store in new column
tqdm.pandas()
df["text_hinglish"] = df["text"].progress_apply(translate)

# Save translated version
df.to_csv("goemotions_train_hinglish.csv", index=False)
print("✅ Translated and saved to goemotions_train_hinglish.csv")
