# Emotion VibeCheck – Hinglish + English Emotion Detection  

An end-to-end pipeline for detecting emotions in **Hinglish (Hindi + English mix)** and English text.  
This project includes dataset preparation, model training, evaluation, and a Flask-based web app for real-time predictions.  

---

## 🚀 Features
- Fine-tuned **GoEmotions-based model** with additional Hinglish samples  
- Training pipeline (`train_emotion.py`) for reproducibility  
- Pre-split dataset (`train.json`, `validation.json`, `test.json`)  
- Flask web app for live predictions with text or screenshots  
- Feedback logging system to improve model iteratively  

---

## 📂 Repository Structure
ai-vibecheck-310/
│
├── dataset/ # JSON dataset files (train.json, validation.json, test.json)
│
├── emotion-web-app/ # Flask app
│ ├── app.py
│ ├── requirements.txt
│ ├── templates/
│ │ └── index.html
│ └── README.md
| ├── Procfile
| ├──.gitattributes
│
├── train_emotion.py # Main training script
├── finetune_emotion_model.py
├── train_model.py
├── predict_emotion.py # Run single text prediction
├── test_emotion.py # Evaluate on test set
├── text_emotion.py # CLI emotion predictions
├── screenshot_emotion.py # Extract + predict from screenshots
├── clean_dataset.py # Cleaning scripts for dataset
├── prepare_dataset.py # Preprocessing scripts
├── translate_to_hinglish.py # Synthetic Hinglish data generator
│
└── README.md # Project documentation (this file)

---

## 📊 Dataset
**Hinglish GoEmotions — Dataset**

- **Languages**: English + Hinglish  
- **Labels**: anger, sadness, joy, fear, surprise, disgust, neutral, love, curiosity, confusion  
- **Splits**: train / validation / test  
- **Format**: JSON (text + label)  
- **Author**: Jagrit Chaudhry  
- **License**: CC-BY-SA 4.0  

---

### Owner

👤 Author
Jagrit Chaudhry
B.Tech CSE, DTU (3rd Year)

https://github.com/Jagrit-09
https://huggingface.co/Hostileic
https://www.linkedin.com/in/jagrit-chaudhry-448690309/

## 📜 License

This project is released under the **CC-BY-SA 4.0 License**.  
You are free to **use, share, and adapt** it, as long as proper credit is given.  

---

## 📚 Citation

If you find this project useful in your research, learning, or applications, please consider citing it.  
It really helps me as a student to grow this project further. 🙌  

**APA:**  
Chaudhry, J. (2025). *Emotion VibeCheck – Hinglish + English Emotion Detection*. GitHub & Hugging Face.  

**BibTeX:**  
```bibtex
@misc{chaudhry2025emotionvibecheck,
  author       = {Chaudhry, Jagrit},
  title        = {Emotion VibeCheck – Hinglish + English Emotion Detection},
  year         = {2025},
  howpublished = {\url{https://github.com/Jagrit-09/emotion-vibecheck}},
}
