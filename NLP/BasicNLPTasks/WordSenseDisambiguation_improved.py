

import nltk
from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk


nltk.download("averaged_perceptron_tagger_eng")

# define an example sentence

sentences = ["I go to the bank to deposit money",
             "The river bank was flooded after the heavy rain"]

word = "bank"

for s in sentences:
    
   print()
    
   print(f"Sentence: {s}")
   
   
   sense_simple_lesk = simple_lesk(s, word)
   print(f"Sense Simple: {sense_simple_lesk.definition()}")
   
   
   sense_adapted_lesk = adapted_lesk(s, word)
   print(f"Sense Adapted: {sense_adapted_lesk.definition()}")
   
   
   sense_cosine_lesk = cosine_lesk(s, word)
   print(f"Sense Cosine: {sense_cosine_lesk.definition()}")
   
   
   
   # print(f"Word: {word}")
   # print(f"Sense: {sense2.definition()}")


# Normally, the cosine_lesk method is the most powerful one when it comes to sense disambiguation.
# Since our sentences are not complicated enough, simple and adapted lesk methods 
# show supremacy compared to the cosine lesk algorithm.


"""
Sentence: I go to the bank to deposit money
Sense Simple: a financial institution that accepts deposits and channels the money into lending activities
Sense Adapted: a financial institution that accepts deposits and channels the money into lending activities
Sense Cosine: a container (usually with a slot in the top) for keeping money at home


Sentence: The river bank was flooded after the heavy rain
Sense Simple: sloping land (especially the slope beside a body of water)
Sense Adapted: sloping land (especially the slope beside a body of water)
Sense Cosine: a supply or stock held in reserve for future use (especially in emergencies)

"""














