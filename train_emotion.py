import os
from datasets import load_dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Paths
dataset_dir = "./dataset"
train_file = os.path.join(dataset_dir, "train.json")
val_file = os.path.join(dataset_dir, "validation.json")
test_file = os.path.join(dataset_dir, "test.json")

# Load dataset
dataset = {
    "train": load_dataset("json", data_files=train_file, split="train"),
    "validation": load_dataset("json", data_files=val_file, split="train"),
    "test": load_dataset("json", data_files=test_file, split="train")
}

# Get unique emotion labels
unique_labels = sorted(set(example["emotion"] for example in dataset["train"]))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Encode labels into numerical form
def encode_labels(example):
    example["label"] = label2id[example["emotion"]]
    return example

dataset = {k: v.map(encode_labels) for k, v in dataset.items()}

# Tokenizer
model_path = "./emotion-model" if os.path.exists("./emotion-model") else "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = {k: v.map(tokenize, batched=True) for k, v in dataset.items()}

# Model
model = XLMRobertaForSequenceClassification.from_pretrained(
    model_path if os.path.exists("./emotion-model") else "xlm-roberta-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Evaluate on test set
results = trainer.evaluate(eval_dataset=dataset["test"])
print("📊 Test Results:", results)

# Save model
model.save_pretrained("./emotion-model")
tokenizer.save_pretrained("./emotion-model")
print("✅ Model saved to ./emotion-model")
