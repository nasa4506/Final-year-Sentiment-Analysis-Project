import json

with open("g:/Final year project/experiments/test.ipynb", "r") as f:
    nb = json.load(f)

# Check if Step 8 is already there to avoid duplicates
if not any("Step 8" in "".join(c.get("source", [])) for c in nb.get("cells", [])):
    new_cells = [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 8: Hindi Robustness Test\n",
        "To prove cross-lingual transfer learning, we will evaluate the model against our reserved `dataset/robust_6_emotions/val.csv`. This split contains the Kaggle Hindi dataset rows that the model has never seen before."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "print(\"Loading Validation Data (which contains the raw Hindi data)...\")\n",
        "val_df = pd.read_csv(\"../dataset/robust_6_emotions/val.csv\")\n",
        "print(f\"Loaded {len(val_df)} total validation samples.\\n\")\n",
        "\n",
        "y_true_hi = []\n",
        "y_pred_hi = []\n",
        "\n",
        "model.eval()\n",
        "print(\"Evaluating (this may take a minute)...\")\n",
        "for i in range(len(val_df)):\n",
        "    text = str(val_df.iloc[i][\"text\"])\n",
        "    true_label_id = int(val_df.iloc[i][\"label\"])\n",
        "    \n",
        "    inputs = tokenizer(text, return_tensors=\"pt\", max_length=128, truncation=True, padding=True).to(device)\n",
        "    with torch.no_grad():\n",
        "        outputs = model(**inputs)\n",
        "        pred_idx = torch.argmax(outputs.logits, dim=-1).item()\n",
        "        \n",
        "    y_true_hi.append(true_label_id)\n",
        "    y_pred_hi.append(pred_idx)\n",
        "    \n",
        "    if (i+1) % 500 == 0:\n",
        "        print(f\"  Processed {i+1} samples...\")\n",
        "\n",
        "print(\"\\n--- HINDI (and some English) VALIDATION CLASSIFICATION REPORT ---\")\n",
        "print(classification_report(y_true_hi, y_pred_hi, target_names=labels_list, zero_division=0))\n",
        "\n",
        "cm_hi = confusion_matrix(y_true_hi, y_pred_hi)\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm_hi, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_list, yticklabels=labels_list)\n",
        "plt.xlabel('Predicted Emotion')\n",
        "plt.ylabel('True Emotion')\n",
        "plt.title('Confusion Matrix - Validation Set (Including Hindi)')\n",
        "plt.show()"
       ]
      }
    ]
    
    nb["cells"].extend(new_cells)
    
    with open("g:/Final year project/experiments/test.ipynb", "w") as f:
        json.dump(nb, f, indent=4)
