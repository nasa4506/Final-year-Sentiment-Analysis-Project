import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# 1. Configuration
MODEL_NAME = "l3cube-pune/hindi-bert-v2"  # Base Hindi BERT
OUTPUT_DIR = "./models_weights/hindi_sentiment"
DATASET_DIR = "./dataset"
MAX_LENGTH = 128  # Truncate to save VRAM on RTX 3050 (4GB)

# Verify GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Load Datasets
print("\nLoading datasets from CSVs...")

# Load Amazon Reviews Dataset
amazon_data_files = {
    "train": os.path.join(DATASET_DIR, "hi_sentiment_train.csv"),
    "val": os.path.join(DATASET_DIR, "hi_sentiment_val.csv")
}
amazon_dataset = load_dataset("csv", data_files=amazon_data_files)

# Load Kaggle Emotions Dataset
kaggle_data_files = {
    "train": os.path.join(DATASET_DIR, "kaggle_hi_sentiment_train.csv"),
    "val": os.path.join(DATASET_DIR, "kaggle_hi_sentiment_val.csv")
}
kaggle_dataset = load_dataset("csv", data_files=kaggle_data_files)

from datasets import concatenate_datasets

# Merge them together!
dataset = {
    "train": concatenate_datasets([amazon_dataset["train"], kaggle_dataset["train"]]),
    "val": concatenate_datasets([amazon_dataset["val"], kaggle_dataset["val"]])
}

# Shuffle the combined datasets so the model doesn't just see all Amazon reviews then all Kaggle reviews
dataset["train"] = dataset["train"].shuffle(seed=42)
dataset["val"] = dataset["val"].shuffle(seed=42)

print(f"Total Combined Training Samples: {len(dataset['train'])}")
print(f"Total Combined Validation Samples: {len(dataset['val'])}")

# Optional: For prototyping/testing the script, use a tiny subset of the data
# Uncomment the following lines if you want to do a quick 2-minute test run
# dataset["train"] = dataset["train"].select(range(1000))
# dataset["val"] = dataset["val"].select(range(200))

# 3. Load Tokenizer & Model
print(f"\nLoading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,  # Negative(0), Neutral(1), Positive(2)
    id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
    label2id={"Negative": 0, "Neutral": 1, "Positive": 2}
)

# 4. Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

print("Tokenizing combined dataset...")
# Since we converted it to a dictionary, we map each split individually
tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_val = dataset["val"].map(tokenize_function, batched=True)

# 5. Metrics Calculation
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6. Training Arguments (Optimized for 4GB VRAM RTX 3050)
print("\nConfiguring Training Arguments for RTX 3050 (4GB)...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save the model at the end of each epoch
    learning_rate=2e-5,
    num_train_epochs=3,           # Typical for fine-tuning
    weight_decay=0.01,
    
    # --- VRAM Optimizations ---
    per_device_train_batch_size=4,   # Very small batch size to fit in VRAM
    per_device_eval_batch_size=4,    
    gradient_accumulation_steps=4,   # Accumulate 4 steps -> Effective Batch Size = 16
    fp16=True,                       # Mixed precision (Cuts VRAM usage heavily)
    # --------------------------
    
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,     # Keep the best epoch model
)

# 7. Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# 8. Train!
print("\nStarting Training...")
trainer.train()

# 9. Save final optimized model
print(f"\nSaving final fine-tuned model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
