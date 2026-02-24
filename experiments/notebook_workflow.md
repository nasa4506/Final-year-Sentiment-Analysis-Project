# Robust Multilingual Emotion Classification Workflow

This document summarizes the end-to-end process executed in `test.ipynb` to solve the catastrophic overfitting issue (where accuracy dropped from 100% to 14% on real-world data).

## 1. The Core Problem
The previous model was trained on a synthetic, templated dataset. It memorized sentence structure rather than learning linguistic emotional variance. To fix this, we transitioned to a highly varied, real-world multilingual dataset and collapsed the taxonomy down to 6 core emotions (Sadness, Joy, Love, Anger, Fear, Surprise).

## 2. Model & Architecture Choices
- **Base Model:** `xlm-roberta-base`
  - *Why:* It is pre-trained on 100 languages (including Hindi and English) and excels at cross-lingual transfer learning.
- **Fine-Tuning Strategy:** LoRA (Low-Rank Adaptation)
  - *Why:* Full fine-tuning of 270M parameters would crash a 4GB VRAM GPU. LoRA freezes the base model and injects tiny trainable rank-decomposition matrices, training only ~0.3% of the weights (approx. 890k parameters) while achieving near full-finetuning performance.
- **Handling Data Imbalance:** Class Weights via PyTorch `CrossEntropyLoss`
  - *Why:* Real-world data is heavily skewed (e.g., lots of Joy, very little Surprise). Without weights, the model would simply guess the majority class. We calculated balanced multipliers and injected them into a Custom Trainer.

## 3. The Datasets Used

### Training Dataset (Steps 2-6)
- **Prepared Source Location:** `experiments/robust_6_emotions/train.csv` (Copied from the main `dataset` split for archival testing)
- **Composition:** 19,424 samples containing a highly shuffled blend of English data (from HuggingFace `dair-ai/emotion`) and Hindi data (from Kaggle `hindi-sentiment-dataset`).

### Final Evaluation - English Only (Step 7)
- **Source:** HuggingFace `dair-ai/emotion` (Test Split)
- **Composition:** 2,000 purely unseen English real-world samples.
- **Purpose:** To verify that the architecture actually learned emotional extraction on standard benchmark data without any data leakage from the training phase.

### Hindi Robustness Test (Step 8)
- **Prepared Source Location:** `experiments/robust_6_emotions/val.csv`
- **Composition:** 2,857 samples containing the reserved validation split. Crucially, this split contains the **unseen Kaggle Hindi rows**.
- **Purpose:** To explicitly prove that our `xlm-roberta` model achieved true cross-lingual transfer learning. By evaluating on this set, we can generate a Classification Report and Confusion Matrix that proves the model accurately classifies real-world Hindi sentences it was never trained on.

## 4. Execution Pipeline
1. **Load Data:** Imported the 19.4k robust train set and mapped to 6 integer classes.
2. **Compute Class Weights:** Calculated the inverse frequency multipliers for all 6 classes to prevent minority class death.
3. **LoRA Injection:** Loaded `xlm-roberta-base`, applied `LoraConfig` (r=8, alpha=16), and mapped the tokenizer.
4. **Custom Trainer:** Overwrote the standard HuggingFace Trainer `compute_loss` method to apply our Class Weights tensor. Tracked `f1_macro` instead of raw accuracy.
5. **Training (VRAM Optimized):** Trained for 3 epochs with FP16 mixed precision, Batch Size 8, and Gradient Accumulation 4. Saved weights to `models_weights/multilingual_6_emotions_lora`.
6. **Evaluation 1:** Pulled the English benchmark test set (`dair-ai/emotion`) and plotted the Confusion Matrix.
7. **Evaluation 2:** Loaded the reserved blended Hindi `val.csv` split, ran inference, and plotted the multilingual Confusion Matrix.
