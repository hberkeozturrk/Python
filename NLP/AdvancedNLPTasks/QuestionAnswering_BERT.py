


from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")


# BERT model that was fine-tuned on squad dataset 
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
 

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)


# fine-tuned bert model for question answering
model = BertForQuestionAnswering.from_pretrained(model_name)


# Function guesses the answers
def predict_answer(context, question):
    
    """
    context = metin
    question = soru
    
    Amaç: metin içerisinden soruyu bulmak
    
    1) tokenize
    2) metnin içerisinde soruyu ara
    3) metnin içerisinde sorunun cevabının nerelerde olabileceğinin skorlarını döndürdü.
    4) skorlardan tokenların indeksleri hesapla
    5) tokenları bul
    6) okunabilirlik için tokenlardan string e çevirdik  
    
    """
    
    
    
    # convert tokens into text and fit the model
    encoding = tokenizer.encode_plus(question, context, return_tensors = "pt", max_length = 512, truncation= True)
    
    # prepare the input tensors
    input_ids = encoding["input_ids"] # toke ids
    attention_mask = encoding["attention_mask"] # specifies which tokens will be paid attention
    
    # run the model and calculate the score
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask = attention_mask, return_dict = False)
        

    # Calculate the start and end indexes that have the highest prob 
    start_idx = torch.argmax(start_scores, dim = 1).item() # beginning idx
    end_idx = torch.argmax(end_scores, dim = 1).item() # end idx

    # By using token ids, obtain the answer text
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_idx: end_idx + 1])

    # merge tokens and make it readable
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer  
    

question = "What is the capital of France?"
context = "France, officially the French Republic, is a country whose capital is Paris."


answer = predict_answer(context, question)
print("Answer:", answer)
















