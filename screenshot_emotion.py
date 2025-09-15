import pytesseract
from PIL import Image
import cv2
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Set up Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Load your fine-tuned model ---
model_path = "./emotion-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
label_map = [
    "anger", "confusion", "curiosity", "fear", "gratitude",
    "joy", "love", "neutral", "sadness", "surprise"
]

# --- Load image and extract text ---
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text.strip()

# --- Predict emotion ---
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_label = torch.argmax(probs).item()
    confidence = probs[0][pred_label].item()
    return label_map[pred_label], confidence

# --- Main flow ---
if __name__ == "__main__":
    image_path = input("📷 Enter the screenshot file path: ")

    if not os.path.exists(image_path):
        print("❌ File not found.")
        exit()

    print("\n🔍 Extracting text from image...")
    text = extract_text_from_image(image_path)
    print("📄 Extracted Text:\n", text)

    if not text:
        print("⚠️ No readable text found.")
    else:
        print("\n🤖 Detecting emotion...")
        emotion, confidence = predict_emotion(text)
        print(f"\n🧠 Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
