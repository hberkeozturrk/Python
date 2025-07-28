

import nltk
from nltk.wsd import lesk 


# download the necessary nltk packages
nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")


# first sentence
s1 = "I go to the bank to deposit money."
w1 = "bank"
sense1 = lesk(nltk.word_tokenize(s1), w1)

print(f"Sentence: {s1}")
print(f"Word: {w1}")
print(f"Sense: {sense1.definition()}")

"""
Sentence: I go to the bank to deposit money.
Word: bank
Sense: a container (usually with a slot in the top) for keeping money at home

"""

# second sentence
s2 = "The river bank was flooded after the heavy rain."
w2 = "bank"
sense2 = lesk(nltk.word_tokenize(s2), w2)

print(f"Sentence: {s2}")
print(f"Word: {w2}")
print(f"Sense: {sense2.definition()}")


"""
Sentence: The river bank is flooded after the heavy rain.
Word: bank
Sense: a container (usually with a slot in the top) for keeping money at home

"""














