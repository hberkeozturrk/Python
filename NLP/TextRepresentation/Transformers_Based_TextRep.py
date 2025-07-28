

# import libs
from transformers import AutoTokenizer, AutoModel
import torch


# load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)



# define text
text = "Transformers can be used for natural language processing"


# tokenize text
inputs = tokenizer(text, return_tensors = "pt") # return as pytorch tensor


# text rep by using model
with torch.no_grad(): # stop calculating the gradians, thereby effective memory usage is provided                 
    outputs = model(**inputs) 



# get last hidden state from model's output
last_hidden_state = outputs.last_hidden_state


# get first token's embedding
first_token_embedding = last_hidden_state[0,0,:].numpy()
print(f"Metin temsili: {first_token_embedding} ")

# print values







