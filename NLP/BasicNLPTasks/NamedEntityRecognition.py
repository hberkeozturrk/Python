

"""
Names entity recognition (NER): text (sentence) -> define named entities 

"""


# import libs
import pandas as pd
import spacy




# with Spacy model, define named entity 
nlp = spacy.load("en_core_web_sm") # spacy lib eng language model

content = "Alice works at Amazon and lives in London. She visited the Biritish Museum last weekend."

doc = nlp(content) # this operation analyzes named entities


for ent in doc.ents:
    
    # ent.text: entity name
    # ent.start_char and ent.end_char, entitiy's first and last characters in text
    # ent.label: entity type
    
    # print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print(ent.text, ent.label_)


entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]


# convert entity list into pandas df

df = pd.DataFrame(entities, columns = ["text", "type", "lemma"])












