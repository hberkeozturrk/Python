

# import libs
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


# create a sample document
document = ["Köpek çok tatlı bir hayvandır,",
            "Köpek ve kuşlar çok tatlı hayvanlardır",
            "Inekler süt üretirler."]



# define vectorizer
tfidf_vectorizer = TfidfVectorizer()


# convert texts into numerical vals
X = tfidf_vectorizer.fit_transform(document)


# examine the word set 
feature_names = tfidf_vectorizer.get_feature_names_out()


# examine the vector rep
vector_rep = X.toarray()

print(f"tf-idf: {vector_rep}")

df_tfidf = pd.DataFrame(vector_rep, columns = feature_names)


# look at avg tf-idf
tf_idf = df_tfidf.mean(axis = 0)


















