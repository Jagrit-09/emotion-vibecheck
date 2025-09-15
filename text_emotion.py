from transformers import pipeline

# Load the emotion classification model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Define emoji for each emotion
EMOJI_MAP = {
    "joy": "😄",
    "anger": "😡",
    "sadness": "😢",
    "love": "❤️",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐"
}

# Function to analyze emotion
def get_text_emotion(text):
    results = classifier(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_results[0]
    return {
        "label": top_emotion['label'],
        "score": round(top_emotion['score'], 2),
        "all": sorted_results
    }

# Run the program
if __name__ == "__main__":
    text = input("Enter text to analyze emotion: ")
    result = get_text_emotion(text)
    emotion = result['label']
    emoji = EMOJI_MAP.get(emotion.lower(), "")
    print(f"Emotion: {emotion} {emoji} (Confidence: {result['score']})")
