

# import libs
from sklearn.feature_extraction.text import CountVectorizer

# create sample text
documents = ["Bu çalışma NGram çalışmasıdır.",
             "Bu çalışma doğal dil işleme çalışmasıdır."]



# uni-, bi- and trigram implementation
vectorizer_unigram = CountVectorizer(ngram_range= (1, 1))
vectorizer_bigram = CountVectorizer(ngram_range= (2, 2))
vectorizer_trigram = CountVectorizer(ngram_range= (3, 3))


# Unigram
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()


# Bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()


# Trigram
X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()


# results evaluation
print(f"unigram features: {unigram_features}")
print(f"bigram features: {bigram_features}")
print(f"trigram features: {trigram_features}")





















