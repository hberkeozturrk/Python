

# import the count vectorizer 
from sklearn.feature_extraction.text import CountVectorizer


# create a dataset
documents = ["kedi bahçede, ama kedi evde değil.",
             "kedi evde"]


# doc = {"s1": "kedi bahçede", "s2": "kedi evde"}


# define the vectorizer
vectorizer = CountVectorizer()


# convert text into numeric vectors
X = vectorizer.fit_transform(documents)


# create the word set
feature_names = vectorizer.get_feature_names_out() 
print(f"Word set: {feature_names}")

# vector representation
vector_rep = X.toarray()


print(f"Vector representation: {vector_rep}")


"""
Word set: ['bahçede' 'evde' 'kedi']
Vector representation: [[1 0 1]
                        [0 1 1]]

"""





