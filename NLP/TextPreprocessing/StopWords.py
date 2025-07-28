
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords") # Dataset which contains most used stop words in different languages


# English stop words analysis (nltk)
stop_words_eng = set(stopwords.words("english"))


# Example eng text
text = "There are some examples of handling stopwords from some texts."
text_list = text.split()

# If the word is not in the stopwords list, add this word into the filtered list.
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered_words: {filtered_words}")


# %% Turkish stop words analysis (nltk)
stopwords_tr = set(stopwords.words("turkish"))

metin = "Merhaba arkadaşlar çok güzel bir ders işliyoruz. Bu ders faydalı mı?"
metin_list = metin.split()

filtered_words_tr = [word for word in metin_list if word.lower() not in stopwords_tr]

print(f"filtered words: {filtered_words_tr}")

# Stop words filtering without using a lib

# Create a stopwords list
tr_stopwords = ["için", "bu", "ile", "mu", "mi", "özel"]


metin = "Bu bir denemedir. Amacımız bu metinde bulunan özel karakterleri elemek mi acaba?"


filtered_words = [word for word in metin.split() if word.lower() not in tr_stopwords]
filtered_stop_words = [word for word in metin.split() if word.lower() in tr_stopwords]

print(f"filtered words: {filtered_words}")











