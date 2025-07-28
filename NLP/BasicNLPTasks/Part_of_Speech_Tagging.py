

import spacy


nlp = spacy.load("en_core_web_sm") # spacy lib eng language model


sentence1 = "What is the weather like today or tomorrow?"
doc1 = nlp(sentence1)

sentence2 = "I have been reading a book lately, and it is very intriguing"
doc2 = nlp(sentence2)


for token in doc1:
    print(token.text, token.pos_)

print()

for token in doc2:
    print(token.text, token.pos_)




