import os
import torch
import numpy as np
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

# 1. Configuration
MODEL_NAME = "xlm-roberta-base"  # Multilingual base model
OUTPUT_DIR = "./models_weights/multilingual_13_emotions_lora"
DATASET_DIR = "./dataset/13_emotions"
MAX_LENGTH = 128  # Keep sequence short for VRAM

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Load Label Map
print("\nLoading Label Mapping...")
with open(os.path.join(DATASET_DIR, "label_mapping.json"), "r") as f:
    id2label = json.load(f)
    # JSON loads keys as strings, convert to ints
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
print(f"Loaded {num_labels} classes: {list(id2label.values())}")

# 3. Load Dataset
print("\nLoading mapped datasets...")
data_files = {
    "train": os.path.join(DATASET_DIR, "train.csv"),
    "val": os.path.join(DATASET_DIR, "val.csv")
}
dataset = load_dataset("csv", data_files=data_files)

# 4. Load Tokenizer & Base Model
print(f"\nLoading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# 5. Apply LoRA (Low-Rank Adaptation)
print("\nInjecting LoRA Adapters into Model...")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,               # Rank of the LoRA matrices
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Which attention weights to adapt 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # This will visually verify we are only training < 1% of the model!

# 6. Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

print("\nTokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 7. Metrics Calculation
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {"accuracy": acc, "f1_weighted": f1}

# 8. Optimized Training Arguments FOR RTX 3050 4GB
print("\nConfiguring Training Arguments for RTX 3050 (4GB VRAM with LoRA)...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  
    save_strategy="epoch",                
    learning_rate=2e-4,              # LoRA typically uses a higher learning rate
    num_train_epochs=3,           
    weight_decay=0.01,
    
    # --- Extreme VRAM Optimizations ---
    per_device_train_batch_size=4,   # Fits in 4GB easily when paired with LoRA
    per_device_eval_batch_size=4,    
    gradient_accumulation_steps=8,   # 4 batch * 8 accum = 32 effective batch size
    fp16=True,                       # Mixed precision (Vital for VRAM)
    # ----------------------------------
    
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,     
)

# 9. Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    compute_metrics=compute_metrics,
)

# 10. Train!
print("\nStarting LoRA Multi-Lingual Emotion Training...")
trainer.train()

# 11. Save
print(f"\nSaving final LoRA model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR) # Saves ONLY the tiny LoRA adapters, not the whole 500MB model!
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! You've trained a 13-class multilingual model!")
