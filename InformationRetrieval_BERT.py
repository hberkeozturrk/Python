


# import libs
from transformers import BertTokenizer, BertModel

import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity


# tokenizer and model create
model_name = "bert-base-uncased" # small-sized BERT model
tokenizer = BertTokenizer.from_pretrained(model_name) # load tokenizer

model = BertModel.from_pretrained(model_name) # BERT model that was previously trained


# create the data: create the compared documents and query sentences 

documents = ["Machine Learning is a field of artificial intelligence",
             "Natural language processing involves understanding human language",
             "Artificial intelligence encompasses machine learning and natural language processing",
             "Deep learning is a subset of machine learning",
             "Data science combines statistics data analysis and machine learning",
             "I go to shop"]


query = "What is deep learning?"



# information retrieval with BERT
def get_embedding(text):
    # tokenize
    inputs = tokenizer(text, return_tensors= "pt", truncation = True, padding = True)
    
    # run the model
    outputs = model(**inputs) # unpacking operation

    # last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # create text rep
    embedding = last_hidden_state.mean(dim = 1)
    
    # return vector as numpy 
    return embedding.detach().numpy()

    
# get embedding vecs for documents and query
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)


# with cosine similarity, calculate the similarity between documents
similarities = cosine_similarity(query_embedding, doc_embeddings)
    

# print each document's similarity score
for i, score in enumerate(similarities[0]):
    print(f"Document: {documents[i]}: \n{score}")


"""

# Machine learning query:
    
Document: Machine Learning is a field of artificial intelligence: 
0.7525447607040405

Document: Natural language processing involves understanding human language: 
0.6778316497802734

Document: Artificial intelligence encompasses machine learning and natural language processing: 
0.7277357578277588

Document: Deep learning is a subset of machine learning: 
0.7297888994216919

Document: Data science combines statistics data analysis and machine learning: 
0.7086363434791565

Document: I go to shop: 
0.5108955502510071

# Natural Language Processing query:

Document: Machine Learning is a field of artificial intelligence: 
0.726584255695343

Document: Natural language processing involves understanding human language: 
0.8367241024971008

Document: Artificial intelligence encompasses machine learning and natural language processing: 
0.7838457226753235

Document: Deep learning is a subset of machine learning: 
0.7001164555549622

Document: Data science combines statistics data analysis and machine learning: 
0.6888637542724609

Document: I go to shop: 
0.4784521460533142


"""

most_similar_index = similarities.argmax()
print(f"Most similar document: {documents[most_similar_index]}")















