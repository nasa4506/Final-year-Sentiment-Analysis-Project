import kagglehub
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

# Our 3 standard sentiments
SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Mapping the 13 emotions to our 3 sentiment classes
EMOTION_TO_SENTIMENT = {
    # Positive (2)
    "joy": 2, "love": 2, "happiness": 2, "amusement": 2, "admiration": 2, "optimism": 2, "excitement": 2, "relief": 2, 
    "positive": 2,
    
    # Neutral (1)
    "neutral": 1, "surprise": 1, "curiosity": 1, "realization": 1, "confusion": 1,
    
    # Negative (0)
    "sadness": 0, "anger": 0, "fear": 0, "disgust": 0, "annoyance": 0, "disappointment": 0, "grief": 0, "remorse": 0, "nervousness": 0,
    "negative": 0
}

def map_emotion_to_sentiment(emotion_label):
    emotion = str(emotion_label).lower().strip()
    return EMOTION_TO_SENTIMENT.get(emotion, 1) # Default to neutral if unknown

def process_kaggle_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("praths71018/hindi-sentiment-dataset")
    
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        print("No CSV files found!")
        return
        
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        all_data.append(df)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # The columns are 'label' and 'sentence'
    # We need to map 'label' to our integer codes (0, 1, 2) and rename 'sentence' to 'text'
    print(f"Loaded {len(combined_df)} rows. Mapping emotions...")
    print(f"Unique emotions in dataset: {combined_df['label'].unique()}")
    
    combined_df['text'] = combined_df['sentence']
    combined_df['label'] = combined_df['label'].apply(map_emotion_to_sentiment)
    
    # Drop rows with empty text
    combined_df.dropna(subset=['text'], inplace=True)
    
    # Keep only what we need
    final_df = combined_df[['text', 'label']]
    
    # We need to split this into train (80%) and validation/test (20%)
    train_df, val_df = train_test_split(final_df, test_size=0.2, random_state=42, stratify=final_df['label'])
    
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "kaggle_hi_sentiment_train.csv")
    val_path = os.path.join(output_dir, "kaggle_hi_sentiment_val.csv")
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    
    print(f"\nSaved {len(train_df)} Kaggle training samples to {train_path}")
    print(f"Saved {len(val_df)} Kaggle validation samples to {val_path}")
    
    print("\nDataset ready!")

if __name__ == "__main__":
    process_kaggle_dataset()
