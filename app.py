from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
import pytesseract
from PIL import Image

# Load model and tokenizer
model_path = "c:/Users/Jagrit/ai-vibecheck-310/emotion-model"  # Adjust if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Get id2label map safely
id2label = model.config.id2label
if not id2label:
    # fallback if id2label is missing
    labels = model.config.label2id
    id2label = {str(v): k for k, v in labels.items()}

# Set up Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        text_input = request.form.get("text_input")
        file = request.files.get("screenshot")
        extracted_text = ""

        if text_input:
            extracted_text = text_input.strip()
        elif file:
            image = Image.open(file.stream)
            extracted_text = pytesseract.image_to_string(image)

        if extracted_text:
            inputs = tokenizer(extracted_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_label_id = torch.argmax(probs, dim=1).item()

                # ✅ Fix for mixed type label keys
                predicted_label = id2label.get(str(predicted_label_id)) or id2label.get(predicted_label_id) or "unknown"
                confidence = probs[0][predicted_label_id].item()

            prediction = {
                "text": extracted_text,
                "emotion": predicted_label,
                "confidence": round(confidence, 2)
            }

    return render_template("index.html", prediction=prediction)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    entry = {
        "text": data.get("text", ""),
        "predicted_emotion": data.get("predicted_emotion", ""),
        "confidence": data.get("confidence", 0),
        "correct": data.get("correct", False),
        "corrected_emotion": data.get("corrected_emotion", None)
    }

    with open("user_feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)