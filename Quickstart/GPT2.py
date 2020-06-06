# First prepare a tokenized input from our text string using 'GPT2Tokenizer'

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Optional
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocab)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "who was Jim Henson ? Jim Henson was a"
indexed_tokens = tokenizer.encode(text)

# convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Use 'GPT2LMHeadModel' to generate the next token following our text.
# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in eval mode
model.eval()

# Put on cuda
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
    print(predictions.size())

# get the predicted next sub-word
predicted_index = torch.argmax(predictions[0,-1,:]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)


