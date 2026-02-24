import os
import glob
import pandas as pd
import kagglehub
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def build_robust_dataset():
    # ---------------------------------------------------------
    # 1. 6 Core Emotions Taxonomy
    # ---------------------------------------------------------
    # We follow the standard dair-ai/emotion taxonomy:
    # 0: sadness
    # 1: joy
    # 2: love
    # 3: anger
    # 4: fear
    # 5: surprise
    
    LABEL_MAP = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    # Mapping granular Kaggle Hindi emotions to our 6 core buckets
    HINDI_TO_CORE_MAP = {
        "joy": 1, "happiness": 1, "amusement": 1, "optimism": 1, "excitement": 1, "relief": 1, "positive": 1,
        "sadness": 0, "grief": 0, "disappointment": 0, "remorse": 0, "negative": 0,
        "anger": 3, "annoyance": 3, "disgust": 3,
        "fear": 4, "nervousness": 4,
        "love": 2, "admiration": 2,
        "surprise": 5
    }
    # Note: We deliberately exclude 'neutral', 'curiosity', 'realization', 'confusion' as they don't cleanly fit the 6 core emotions.
    
    print("Step 1: Loading English Dataset (dair-ai/emotion) from HuggingFace...")
    eng_dataset = load_dataset("dair-ai/emotion", "split")
    eng_train_df = eng_dataset["train"].to_pandas()
    eng_val_df = eng_dataset["validation"].to_pandas()
    
    # HuggingFace dataset is already 'text' and 'label' (0-5)
    print(f"  English Train: {len(eng_train_df)}, Val: {len(eng_val_df)}")
    
    print("\nStep 2: Loading Hindi Dataset from Kaggle...")
    path = kagglehub.dataset_download("praths71018/hindi-sentiment-dataset")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    hindi_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        hindi_data.append(df)
        
    raw_hindi_df = pd.concat(hindi_data, ignore_index=True)
    raw_hindi_df['text'] = raw_hindi_df['sentence'] # rename
    
    # Apply Mapping
    raw_hindi_df['mapped_emotion'] = raw_hindi_df['label'].str.lower().str.strip()
    raw_hindi_df['label'] = raw_hindi_df['mapped_emotion'].map(HINDI_TO_CORE_MAP)
    
    # Drop rows that didn't map (e.g., 'neutral') and NaN texts
    hindi_df = raw_hindi_df.dropna(subset=['label', 'text'])
    hindi_df['label'] = hindi_df['label'].astype(int)
    hindi_df = hindi_df[['text', 'label']]
    
    print(f"  Hindi valid extracted: {len(hindi_df)}")
    
    # Split Hindi into Train/Val (80/20) using stratification
    hin_train_df, hin_val_df = train_test_split(
        hindi_df, test_size=0.2, random_state=42, stratify=hindi_df['label']
    )
    print(f"  Hindi Train: {len(hin_train_df)}, Val: {len(hin_val_df)}")

    # ---------------------------------------------------------
    # 3. Merge & Shuffle
    # ---------------------------------------------------------
    print("\nStep 3: Merging & Shuffling...")
    final_train_df = pd.concat([eng_train_df, hin_train_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    final_val_df = pd.concat([eng_val_df, hin_val_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Final Robust Train: {len(final_train_df)}")
    print(f"  Final Robust Val: {len(final_val_df)}")
    
    # Class Distribution Output
    print("\nClass Distribution (Train):")
    dist = final_train_df['label'].value_counts().sort_index()
    for class_id, count in dist.items():
        print(f"  {LABEL_MAP[class_id]:<10}: {count}")
    
    # ---------------------------------------------------------
    # 4. Save
    # ---------------------------------------------------------
    output_dir = "dataset/robust_6_emotions"
    os.makedirs(output_dir, exist_ok=True)
    
    final_train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding='utf-8')
    final_val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding='utf-8')
    
    # Save the label mapping for the training script
    import json
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(LABEL_MAP, f, indent=4)
        
    print(f"\nSaved! Output Directory: {output_dir}")

if __name__ == "__main__":
    build_robust_dataset()
