from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load tokenizer and model
model_name = "nateraw/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Get user input
text = input("Enter your sentence: ")

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = softmax(logits.numpy()[0])
predicted_label = probs.argmax()

# Load label map
labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
          'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
          'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
          'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
          'sadness', 'surprise', 'neutral']

print(f"Predicted emotion: {labels[predicted_label]}")
