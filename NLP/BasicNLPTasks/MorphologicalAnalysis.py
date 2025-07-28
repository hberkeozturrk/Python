

import spacy


nlp = spacy.load("en_core_web_sm") # spacy lib eng language model


# examined words

word = "I go to school 123"

doc = nlp(word)


for token in doc:
    print("Text: {token.text}")             # word itself
    print(f"lemma: {token.lemma_}")         # stem part of the word    
    print(f"POS: {token.pos_}")             # grammatical perk of the word
    print(f"Tag: {token.tag_}")             # detailed grammatical perk of the word
    print(f"Dependency: {token.dep_}")      # role of the word
    print(f"Shape: {token.shape_}")         # character type
    print(f"Is alpha: {token.is_alpha}")    # Checks whether the word comprises of alphabetical letter or not 
    print(f"Is stop: {token.is_stop}")      # checks to see if the word is a stopword
    print(f"Morphology: {token.morph}")     # Morphological perks of the word
    print(f"Is Plural: {'Number = Plur' in token.morph}") # checks if the word is plural



    print()

