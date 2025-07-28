

import nltk

nltk.download("wordnet") # Required database for the lemmatization

from nltk.stem import PorterStemmer, SnowballStemmer # Function for stemming


# Porter stemmer object
stemmer_porter = PorterStemmer() # or Snowball Stemmer
stemmer_snowball = SnowballStemmer("english")

words = ["running", "runner", "ran", "runs", "better", "go", "went", "studied"]

# Finding stems of the words
stems_porter = [stemmer_porter.stem(w) for w in words]
print(f"Porter Stemmer: {stems_porter}")


# Finding stems of the words
stems_snowball = [stemmer_snowball.stem(w) for w in words]
print(f"Snowball Stemmer: {stems_snowball}")



 # %% Lemmatization

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


words = ["running", "runner", "ran", "runs", "better", "go", "went", "studies"]

# Finding stems of the words
lemmas = [lemmatizer.lemmatize(w, pos = "v") for w in words]
print(f"Lemma: {lemmas}")











