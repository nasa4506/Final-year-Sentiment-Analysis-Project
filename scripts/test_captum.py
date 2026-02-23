import torch
from backend.services.model_loader import model_loader
from captum.attr import LayerIntegratedGradients

# Load model
model, device, config, tokenizer = model_loader.load_text_model()

text = "मैं यहाँ आकर बहुत उत्साहित और खुश हूँ।"
encoded_input = tokenizer(text, return_tensors='pt')
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

def forward_func(inputs, attention_mask=None):
    return model(input_ids=inputs, attention_mask=attention_mask).logits

# Discover embedding layer
if hasattr(model, 'base_model') and hasattr(model.base_model.model, 'roberta'):
    embedding_layer = model.base_model.model.roberta.embeddings.word_embeddings
elif hasattr(model, 'roberta'):
    embedding_layer = model.roberta.embeddings.word_embeddings
else:
    print("Could not find embedding layer!")
    exit(1)

print(f"Using embedding layer: {embedding_layer}")

lig = LayerIntegratedGradients(forward_func, embedding_layer)

logits = forward_func(input_ids, attention_mask)
target_class = torch.argmax(logits, dim=1).item()
print(f"Target class: {target_class}")

# We need a baseline
baseline = torch.zeros_like(input_ids).to(device)

attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=baseline,
    target=target_class,
    additional_forward_args=(attention_mask,),
    return_convergence_delta=True
)

attributions = attributions.sum(dim=-1).squeeze(0)
attributions = attributions / torch.norm(attributions)
attributions = attributions.cpu().detach().numpy()

tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

for token, attr in zip(tokens, attributions):
    print(f"{token}: {attr:.4f}")
