import os
import torch
import numpy as np
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.utils.class_weight import compute_class_weight
import evaluate
from torch import nn

# 1. Configuration
MODEL_NAME = "xlm-roberta-base"  
OUTPUT_DIR = "./models_weights/multilingual_6_emotions_lora"
DATASET_DIR = "./dataset/robust_6_emotions"
MAX_LENGTH = 128  

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Load Label Map
print("\nLoading Label Mapping...")
with open(os.path.join(DATASET_DIR, "label_mapping.json"), "r") as f:
    id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
print(f"Loaded {num_labels} classes: {list(id2label.values())}")

# 3. Load Dataset
print("\nLoading robust mapped datasets...")
data_files = {
    "train": os.path.join(DATASET_DIR, "train.csv"),
    "val": os.path.join(DATASET_DIR, "val.csv")
}
dataset = load_dataset("csv", data_files=data_files)

# 4. Compute Class Weights to handle Imbalance
print("\nComputing Class Weights...")
train_labels = np.array(dataset["train"]["label"])
unique_classes = np.unique(train_labels)
class_weights_np = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

for c, w in zip(unique_classes, class_weights_np):
    print(f"  {id2label[c]:<10}: {w:.3f} weight")

# 5. Load Tokenizer & Base Model
print(f"\nLoading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# 6. Apply LoRA (Low-Rank Adaptation)
print("\nInjecting LoRA Adapters into Model...")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,               
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# 7. Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

print("\nTokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 8. Metrics Calculation
metric_f1 = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    # We use F1 Macro to equally weigh minority classes like surprise or love
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": acc, "f1_macro": f1_macro}

# 9. Custom Trainer for Class Weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 10. Optimized Training Arguments FOR RTX 3050 4GB
print("\nConfiguring Training Arguments for RTX 3050 (4GB VRAM with LoRA)...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  
    save_strategy="epoch",  
    learning_rate=3e-4,              
    num_train_epochs=3,           
    weight_decay=0.01,
    
    # --- Extreme VRAM Optimizations ---
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    gradient_accumulation_steps=4,   # 8 * 4 = 32 effective batch size
    fp16=True,                       
    # ----------------------------------
    
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

# 11. Setup Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    compute_metrics=compute_metrics,
)

# 12. Train!
print("\nStarting LoRA Multi-Lingual Emotion Training...")
trainer.train()

# 13. Save
print(f"\nSaving final LoRA model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR) 
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! You've successfully trained a highly robust 6-class multilingual emotion classifier!")
