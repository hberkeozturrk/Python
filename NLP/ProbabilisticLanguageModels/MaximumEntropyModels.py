

"""
classification problem: sentiment analysis -> positive or negative classification

"""

# import libs
import nltk
from nltk.classify import MaxentClassifier



# define dataset
# define dataset
train_data = [
    ({"Love": True, "amazing": True, "happy": True, "terrible": False}, "positive"),
    ({"hate": True, "terrible": True}, "negative"),
    ({"joy": True, "happy": True, "hate":False}, "positive"),
    ({"sad": True, "depressed": True, "love": False}, "negative")
    
    ]
              



# train maximum entropy classifier
classifier = MaxentClassifier.train(train_data, max_iter = 10)


# test with new sentence
test_sentence = "I do not love this movie." # I hate this movie and it was terrible.

features = {word: (word in test_sentence.lower().split()) for word in ["love", "amazing", "terrible", "joy", "depressed", "sad", "happy"]}


label = classifier.classify(features)
print(f"Result: {label}")























