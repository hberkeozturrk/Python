
# import libs
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000



# createe hmm
nltk.download("conll2000")

train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

print(f"Train data: {train_data[:1]}")

# create a sentence and label each word 
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# Model needs bigger data to make correct 
test_sentence = "I like going to school".split()


tags = hmm_tagger.tag(test_sentence)








