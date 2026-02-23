import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = "./models_weights/hindi_sentiment"
TEST_FILE = "./dataset/hi_sentiment_test.csv"
KAGGLE_TEST = "./dataset/kaggle_hi_sentiment_val.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading fine-tuned model from {MODEL_DIR} to {device}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have trained the model first.")
    exit(1)

def evaluate_on_file(filepath, name):
    print(f"\nEvaluating on {name} ({filepath})...")
    if not os.path.exists(filepath):
        print("File not found.")
        return
        
    df = pd.read_csv(filepath)
    if 'text' not in df.columns or 'label' not in df.columns:
        print("CSV must have 'text' and 'label' columns.")
        return
        
    df = df.dropna(subset=['text', 'label'])
    
    # Take a sample to evaluate quickly (e.g., 500 samples)
    df = df.sample(min(500, len(df)), random_state=42)
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    
    batch_size = 16
    preds = []
    
    print(f"Running inference on {len(texts)} samples...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds)
            
    # Calculate metrics
    acc = accuracy_score(true_labels, preds)
    print(f"\n{name} Accuracy: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=["Negative", "Neutral", "Positive"]))
    
    # Compare with a tiny dummy naive baseline (e.g. predicting majority class)
    majority_class = df['label'].mode()[0]
    baseline_acc = sum([1 for label in true_labels if label == majority_class]) / len(true_labels)
    print(f"Majority Class Baseline: {baseline_acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate_on_file(TEST_FILE, "Amazon Test Set (Unseen during training)")
    evaluate_on_file(KAGGLE_TEST, "Kaggle Validation Set")
    
