import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "xlm-roberta-base"
LORA_PATH = "./models_weights/multilingual_13_emotions_lora"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=13
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(DEVICE)
model.eval()

with open("./dataset/13_emotions/label_mapping.json", "r") as f:
    id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}

# dair-ai/emotion labels: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
HUGGINGFACE_EMOTION_MAP = {
    0: "sadness",
    1: "joy",
    2: "love", # or admiration
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Load the test set of dair-ai/emotion
print("Loading huggingface dataset dair-ai/emotion...")
dataset = load_dataset("dair-ai/emotion", "split", split="test")

correct = 0
total = 0

print("Evaluating...")
for i in tqdm(range(min(1000, len(dataset)))):
    item = dataset[i]
    text = item["text"]
    true_label_id = item["label"]
    true_emotion = HUGGINGFACE_EMOTION_MAP[true_label_id]
    
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        
    pred_emotion = id2label[pred_idx]
    
    # We will accept exact matches or close matches
    if true_emotion == "joy" and pred_emotion in ["joy", "happiness", "amusement", "excitement", "optimism"]:
        correct += 1
    elif true_emotion == "love" and pred_emotion in ["love", "admiration"]:
        correct += 1
    elif true_emotion == "anger" and pred_emotion in ["anger", "annoyance"]:
        correct += 1
    elif true_emotion == pred_emotion:
        correct += 1
        
    total += 1

accuracy = correct / total
print(f"\nAccuracy on Real-World Data (dair-ai/emotion test set): {accuracy * 100:.2f}%")
print(f"Correct: {correct}, Total: {total}")
