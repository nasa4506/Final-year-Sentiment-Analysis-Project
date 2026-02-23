import pandas as pd
import os

def prepare_13_class_dataset():
    # The dataset provides train, val, and test splits.
    train_path = "g:/Final year project/dataset/multilingual-emotion-dataset/balanced/train.csv"
    val_path = "g:/Final year project/dataset/multilingual-emotion-dataset/balanced/val.csv"
    
    print(f"Loading {train_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Get 13 unique emotions
    emotions = sorted(train_df['emotion'].unique().tolist())
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotions)}
    id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}
    
    print("\nTarget 13 Emotion Mapping:")
    for e, i in emotion_to_id.items():
        print(f"  {i}: {e}")
        
    # Apply mapping
    train_df['label'] = train_df['emotion'].map(emotion_to_id)
    train_df = train_df[['text', 'label']]
    train_df.dropna(subset=['text', 'label'], inplace=True)
    train_df['label'] = train_df['label'].astype(int)
    
    val_df['label'] = val_df['emotion'].map(emotion_to_id)
    val_df = val_df[['text', 'label']]
    val_df.dropna(subset=['text', 'label'], inplace=True)
    val_df['label'] = val_df['label'].astype(int)
    
    # 4. Save to CSV in our target location
    output_dir = "dataset/13_emotions"
    os.makedirs(output_dir, exist_ok=True)
    
    out_train_path = os.path.join(output_dir, "train.csv")
    out_val_path = os.path.join(output_dir, "val.csv")
    
    train_df.to_csv(out_train_path, index=False, encoding='utf-8')
    val_df.to_csv(out_val_path, index=False, encoding='utf-8')
    
    print(f"\nTarget train saved: {len(train_df)} samples")
    print(f"Target val saved: {len(val_df)} samples")
    
    # Save the id2label mapping for the training script
    import json
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(id_to_emotion, f, indent=4)
        
    print("Data preparation complete!")

if __name__ == "__main__":
    prepare_13_class_dataset()
