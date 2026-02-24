import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_robust():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_MODEL = "xlm-roberta-base"
    LORA_PATH = "./models_weights/multilingual_6_emotions_lora"
    
    if not os.path.exists(LORA_PATH):
        print("Model not found! Please train the model first by running scripts/train_robust_emotion.py")
        return

    print(f"Loading 6-Class LoRA Model from {LORA_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=6
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(DEVICE)
    model.eval()

    with open("./dataset/robust_6_emotions/label_mapping.json", "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
    
    # Let's evaluate on dair-ai English test set
    print("\nLoading huggingface dataset dair-ai/emotion (English test set)...")
    dataset = load_dataset("dair-ai/emotion", "split", split="test")

    y_true = []
    y_pred = []

    print("Evaluating English 6-Class Test Set...")
    # For speed in this demo, evaluate first 1000
    for i in range(min(1000, len(dataset))):
        text = dataset[i]["text"]
        true_label_id = dataset[i]["label"]
        
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=-1).item()
            
        y_true.append(true_label_id)
        y_pred.append(pred_idx)
        
        if (i+1) % 200 == 0:
            print(f"  Processed {i+1} samples...")

    labels_list = [id2label[i] for i in range(6)]
    
    print("\n---------------------------------------------------------")
    print("CLASSIFICATION REPORT (English - dair-ai/emotion)")
    print("---------------------------------------------------------")
    print(classification_report(y_true, y_pred, target_names=labels_list))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.title('Confusion Matrix - dair-ai/emotion (Test)')
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\nSaved Confusion Matrix to {cm_path}")

if __name__ == "__main__":
    evaluate_robust()
