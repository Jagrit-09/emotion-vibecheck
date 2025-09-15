from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Load the dataset from JSON files
dataset = load_dataset("json", data_files={
    "train": "dataset/train.json",
    "validation": "dataset/validation.json",
    "test": "dataset/test.json"
})

# Print column names
print("✅ Columns in dataset:", dataset["train"].column_names)

# Get label list from training set
label_list = sorted(list(set(example["emotion"] for example in dataset["train"])))
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}
num_labels = len(label_list)

print("✅ Number of labels:", num_labels)
print("✅ Label mapping:", label_to_id)

# Encode the labels as integers
def encode_labels(example):
    example["label"] = label_to_id[example["emotion"]]
    return example

dataset = dataset.map(encode_labels)

# Load tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Tokenize the dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)

# Set format for PyTorch
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments
training_args = TrainingArguments(
    output_dir="./emotion-model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train the model
print("🧠 Fine-tuning model...")
trainer.train()

# Save the final model
model.save_pretrained("./emotion-model")
tokenizer.save_pretrained("./emotion-model")
print("✅ Model saved to ./emotion-model")
