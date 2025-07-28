
"""
Part of Speech (POS): Kelimelerin uygun sözcük türünü bulma çalışması

ex: I (Zamir) am a teacher (isim)

"""


# import libs
import nltk
from nltk.tag import hmm


# define dataset
train_data = [[("I", "PRP"),("am", "VBP"), ("a", "DT"), ("teacher", "NN")],
              [("You", "PRP"),("are", "VBP"), ("a", "DT"), ("student", "NN")]]


# createe hmm
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# create a sentence and label each word 
test_sentence = "He is a driver".split()
tags = hmm_tagger.tag(test_sentence)
print(f"New sentence: {tags}")


# Model needs bigger data to make correct assumptions
"""
New sentence: [('He', 'PRP'), ('is', 'PRP'), ('a', 'PRP'), ('driver', 'PRP')]

"""