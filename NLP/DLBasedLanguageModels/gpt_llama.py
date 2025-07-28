"""
metin üretimi

gpt-2 metin üretimi çalışması
llama 

"""


# import libs 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM



# define the model
model_name = "gpt2"
model_name_llama = "huggyllama/llama-7b"   # llama



# tokenizer and creating the model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama) # llama


model = GPT2LMHeadModel.from_pretrained(model_name, from_tf = False)
model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama)


# first text for text production
text = "Afternoon,"

# tokenization
inputs = tokenizer.encode(text, return_tensors = "pt")
inputs_llama = tokenizer_llama.encode(text, return_tensors = "pt")


# text produce
outputs = model.generate(inputs, max_length = 55) # inputs = starting point of the model
outputs_llama = model_llama.generate(inputs.input_ids, max_length = 55) # llama



# We need to make readable the tokens model produce 
generated_text = tokenizer.decode(outputs[0], skip_special_tokens= True) # Remove special tokens (start and end tokens) from the text
generated_text_llama = tokenizer.decode(outputs[0], skip_special_tokens= True) # llama

# Print the produced text
print(generated_text)



# create dataset


#  










