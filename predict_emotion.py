import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Load the model and tokenizer
model_path = "./emotion-model"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label mapping
id2label = {
    0: "anger",
    1: "confusion",
    2: "curiosity",
    3: "disgust",
    4: "fear",
    5: "joy",
    6: "love",
    7: "neutral",
    8: "sadness",
    9: "surprise"
}

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("🧠 Hinglish Emotion Detector")
print("Type your sentence below. Type 'q', 'quit', or 'exit' to stop.\n")

while True:
    text = input("You: ")
    if text.lower() in ["q", "quit", "exit"]:
        print("👋 Exiting. Have a good day!")
        break

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
        emotion = id2label[pred]

    print(f"🤖 Predicted Emotion: {emotion} (Confidence: {confidence:.2f})\n")
