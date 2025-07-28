"""
spam veri seti -> spam and ham data -> binary classification (decision tree)

"""

# import libs
import pandas as pd




# import dataset
df = pd.read_csv("metin_siniflandirma_spam_veri_seti.csv", encoding = "latin-1")

# Drop nan values
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)

# Rename columns
df.columns = ["label", "text"]

# EDA (Exploratory Data Analysis) -> Missing values
print(df.isna().sum())


# %% Text cleaning and preprocessing: special characters, lowercase, tokenization, stopwords, lemmatize

import nltk
import re

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords") # most used and meaningless words should be removed
nltk.download("wordnet") # to find lemmas
nltk.download("omw-1.4") # a dataset which contains different language's words and belongs to wordnet 

text = list(df.text)
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    # remove special characters
    r = re.sub("[^a-zA-Z]", " ", text[i])
    
    r = r.lower() # lowercase the letters
    
    r = r.split() # split the words
    
    r = [word for word in r if word not in stopwords.words("english")] # remove the stopwords

    r = [lemmatizer.lemmatize(word) for word in r]

    r = " ".join(r)

    corpus.append(r)    



df["text2"] = corpus


# %% Model training and evaluation

X = df["text2"]
y = df["label"] # target variable

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)


# feature extraction (bag of words)

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)


# classifier model training and evaluation
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train_cv, y_train)

X_test_cv = cv.transform(X_test)


# prediction
prediction = dt.predict(X_test_cv)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, prediction)

accuracy = 100*(sum(sum(conf_mat))- conf_mat[1,0] - conf_mat[0, 1]) / sum(sum(conf_mat))
print(f"Accuracy: {accuracy}")



































