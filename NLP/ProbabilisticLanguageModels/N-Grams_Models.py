

# import libs 
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize # tokenization 
from collections import Counter


# create sample dataset
corpus = ["I love apple",
          "I love him",
          "I love NLP",
          "You love me",
          "He loves apple",
          "They love apple",
          "I love you and you love me"]



""" 
Problem Definition
    Language model
    predicting next word -> text generation (Ngram model)
    
    ex: I ... (Love) ... (apple) 

"""


# tokenize the data
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]


# create bigrams
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

bigrams_freq = Counter(bigrams)


# create trigrams
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

trigram_freq = Counter(trigrams)




# model testing  
# I love bigramından sonra "you" veya "apple" kelimelerinin gelme olasılığı
bigram = ("i", "love") # goal bigram

# "i love you" olma olasılığı
prob_you = trigram_freq[("i", "love", "you")] / bigrams_freq[bigram]
print(f"Prob of you: {prob_you}")


# i love apple olasılığı
prob_apple = trigram_freq[("i", "love", "apple")] / bigrams_freq[bigram]
print(f"Prob of apple: {prob_apple}")
















