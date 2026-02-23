import torch
import numpy as np
from scipy.special import softmax
from backend.services.model_loader import model_loader
from backend.src.config.settings import TEXT_MODEL_CONFIG
from captum.attr import LayerIntegratedGradients

def predict_text_sentiment(text: str):
    """
    Predict sentiment from text using RoBERTa model.
    """
    # Load model and tokenizer
    model, device, config, tokenizer = model_loader.load_text_model()
    
    # Preprocess text
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Inference
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Post-processing
    scores = output.logits[0].detach().cpu().numpy()
    scores = softmax(scores)
    
    # Get ranking
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    # Map to labels
    labels = TEXT_MODEL_CONFIG["labels"]
    
    top_label_idx = ranking[0]
    top_score = scores[top_label_idx]
    sentiment = labels[top_label_idx]
    confidence = float(top_score)
    
    # === Explainable AI (XAI) with Captum ===
    reasoning = []
    try:
        # 1. Provide a wrapper for Captum to call the model
        def forward_func(inputs, attention_mask=None):
            return model(input_ids=inputs, attention_mask=attention_mask).logits

        # 2. Safely find the word embeddings layer (supports Peft/LoRA bounds)
        embedding_layer = None
        if hasattr(model, 'base_model') and hasattr(model.base_model.model, 'roberta'):
            embedding_layer = model.base_model.model.roberta.embeddings.word_embeddings
        elif hasattr(model, 'roberta'):
            embedding_layer = model.roberta.embeddings.word_embeddings
            
        if embedding_layer:
            lig = LayerIntegratedGradients(forward_func, embedding_layer)
            
            # 3. Calculate Token Attributions
            # Note: We must specify `target=top_label_idx` to see why it picked this specific emotion
            attributions = lig.attribute(
                inputs=encoded_input['input_ids'],
                target=int(top_label_idx),
                additional_forward_args=(encoded_input['attention_mask'],),
                internal_batch_size=1
            )
            
            # 4. Summarize and Normalize weights over the hidden embedding dimension
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.cpu().detach().numpy()
            
            # 5. Map weights back to readable text tokens
            tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0].tolist())
            
            # Clean up special tokens (<s>, </s>) and piece together subwords (_)
            for token, weight in zip(tokens, attributions):
                if token not in ["<s>", "</s>", "<pad>"]:
                    # XLM-R uses sentencepiece which marks spaces with U+2581
                    # We will completely remove it since the React UI renders spans with margins anyway.
                    # We also remove '_' just in case it interpreted it via a different encoding.
                    clean_token = token.replace("\u2581", "").replace("_", "")
                    # Only add meaningful weights (ignoring absolute zero padding) and non-empty tokens
                    if abs(weight) > 0.01 and clean_token.strip():
                        reasoning.append({
                            "word": clean_token,
                            "weight": float(weight)
                        })
    except Exception as e:
        print(f"Failed to generate XAI reasoning: {e}")

    return sentiment, confidence, reasoning
