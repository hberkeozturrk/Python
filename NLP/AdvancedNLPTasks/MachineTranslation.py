


from transformers import MarianMTModel, MarianTokenizer


model_name = "Helsinki-NLP/opus-mt-fr-en" # english to french "Helsinki-NLP/opus-mt-fr-en"

tokenizer = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(model_name)


text = "hello, what is your name?"


# make encoding, then give them as an input
translated_text = model.generate(**tokenizer(text, return_tensors = "pt", padding = True))


# translated
translated_text = tokenizer.decode(translated_text[0], skip_special_tokens= True)

print(f"Translated_text: {translated_text}")


