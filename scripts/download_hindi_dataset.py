import pandas as pd
from datasets import load_dataset
import os

# Map Amazon stars (1-5) to our sentiments (0: Negative, 1: Neutral, 2: Positive)
def map_stars_to_sentiment(stars):
    if stars <= 2:
        return 0  # Negative
    elif stars == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

def prepare_hindi_dataset():
    print("Downloading Amazon Reviews Multi (Language: hindi)...")
    
    # The 'amazon_reviews_multi' dataset contains reviews in multiple languages.
    # We specify 'hi' to strictly download the Hindi subset.
    dataset = load_dataset("amazon_reviews_multi", "hi")
    
    # Function to format a single split (train, validation, test)
    def process_split(split_name, hf_split):
        print(f"Processing {split_name} split...")
        df = pd.DataFrame(hf_split)
        
        # We need 'review_body' as text and 'stars' as the label.
        # Create 'text' and 'label' columns
        df['text'] = df['review_body']
        df['label'] = df['stars'].apply(map_stars_to_sentiment)
        
        # Keep only the columns we need
        df = df[['text', 'label']]
        
        # Drop empty reviews
        df.dropna(subset=['text'], inplace=True)
        
        return df

    # Process all splits
    train_df = process_split("train", dataset['train'])
    val_df = process_split("validation", dataset['validation'])
    test_df = process_split("test", dataset['test'])
    
    # Since fine-tuning BERT usually requires a train and a test set, 
    # we can optionally merge train+val, or just use train and val. 
    # For simplicity, we'll save train and test. We can use `val` during training too.
    
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "hi_sentiment_train.csv")
    val_path = os.path.join(output_dir, "hi_sentiment_val.csv")
    test_path = os.path.join(output_dir, "hi_sentiment_test.csv")
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"\nSaved {len(train_df)} training samples to {train_path}")
    print(f"Saved {len(val_df)} validation samples to {val_path}")
    print(f"Saved {len(test_df)} testing samples to {test_path}")
    print("\nDataset ready for PyTorch Training!")

if __name__ == "__main__":
    prepare_hindi_dataset()
