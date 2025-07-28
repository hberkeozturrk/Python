"""
Solve the classification problem in NLP with RNN (Deep Learning Based Language Model)
Sentiment Analysis -> labeling sentences as -> positive or negative

restaurant comments evaluation

"""

# import libs
import numpy as np
import pandas as pd
import tensorflow as tf

from gensim.models import Word2Vec


# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# create dataset

data = {
    "text": [
        "Yemekler harikaydı",
        "Garson çok güler yüzlüydü",
        "Tatlılar nefisti",
        "Ortam çok şıktı",
        "Servis hızlıydı",
        "Yemekler sıcaktı ve tazeydi",
        "Menüde çok fazla seçenek vardı",
        "Fiyatlar gayet uygundu",
        "Müzikler çok hoştu",
        "Rezervasyon sistemi sorunsuz çalıştı",
        "Porsiyonlar doyurucuydu",
        "Sunum çok özenliydi",
        "Tatlar birbirine çok uyumluydu",
        "Masa düzeni çok iyiydi",
        "Baharat dengesi tam yerindeydi",
        "Kahve gerçekten çok iyiydi",
        "Tatlıların sunumu çok hoştu",
        "Çalışanlar çok kibar",
        "Servis elemanları oldukça profesyoneldi",
        "Her şey beklentimin üzerindeydi",
        "Yemekler inanılmaz lezzetliydi",
        "Personel çok yardımseverdi",
        "Tatlar çok uyumluydu",
        "Ortam huzur vericiydi",
        "Servis tam zamanındaydı",
        "Yemekler taze ve sıcaktı",
        "Menüde vegan seçenekler vardı",
        "Fiyat-performans harikaydı",
        "Müzik tam kararındaydı",
        "Rezervasyon kolayca yapıldı",
        "Porsiyonlar yeterince büyüktü",
        "Sunum sanatsal bir görünümdeydi",
        "Tatlar çok dengeliydi",
        "Masalar tertemizdi",
        "Baharatlar çok iyi ayarlanmıştı",
        "Kahvenin aroması harikaydı",
        "Tatlılar hafif ve lezzetliydi",
        "Çalışanlar oldukça nazikti",
        "Servis çok dikkatliydi",
        "Her şey mükemmeldi",
        "Yemekler lezzet doluydu",
        "Çalışanlar çok anlayışlıydı",
        "Her şey zamanında geldi",
        "Sunum çok yaratıcıydı",
        "Fiyatlar oldukça uygun",
        "Kahve taze çekilmişti",
        "Tatlılar ev yapımı gibiydi",
        "Servis çok düzenliydi",
        "Yemeklerin kokusu bile güzeldi",
        "Personel çok nazikti",
        "Yemek çok geç geldi",
        "Garsonlar ilgisizdi",
        "Tatlı bayattı",
        "Restoran çok kalabalıktı",
        "Servis inanılmaz yavaştı",
        "Yemekler soğuktu",
        "Menü çok sıradandı",
        "Fiyatlar çok yüksekti",
        "Müzik çok rahatsız ediciydi",
        "Rezervasyonumu karıştırmışlar",
        "Porsiyonlar çok küçüktü",
        "Sunum özensizdi",
        "Lezzet çok yapaydı",
        "Masalar çok pisti",
        "Baharatlar abartılmıştı",
        "Kahve çok kötüydü",
        "Tatlılar fazla şekerliydi",
        "Çalışanlar kaba davrandı",
        "Servis sürekli gecikiyordu",
        "Hiç memnun kalmadım",
        "Yemek yağ içindeydi",
        "Garson yüzü asıktı",
        "Tatlılar çok ağırdı",
        "Ortam boğucuydu",
        "Servis unutkandı",
        "Yemekler bayattı",
        "Menü karışıktı",
        "Fiyatlar gereksiz pahalıydı",
        "Müzik çok yüksekti",
        "Rezervasyon iptal olmuştu",
        "Porsiyonlar yetersizdi",
        "Sunum vasattı",
        "Tatlar birbirine uymuyordu",
        "Masalar yapışıktı",
        "Baharat yok gibiydi",
        "Kahve beklediğim gibi değildi",
        "Tatlılar tatsızdı",
        "Çalışanlar ilgisizdi",
        "Servis aksıyordu",
        "Yemek çok tuzluydu",
        "Personel yardım etmiyordu",
        "Tatlarda eksiklik vardı",
        "Ortam çok sıkıcıydı",
        "Servis tamamen rezaletti"
        "Yemekten sonra midem bozuldu",
        "Çatal ve bıçaklar kirliydi",
        "Garson siparişi yanlış getirdi",
        "İçerisi aşırı dumanlıydı",
        "Tuvaletler çok pisti",
        "Yemeğin içinde saç çıktı",
        "Masada böcek gördüm, hijyen berbattı"
    ],
    "label": [
        "positive"] * 50 + ["negative"] * 50
}

# converting this dataset into a dataframe
df = pd.DataFrame(data)


# %% text cleaning and preprocessing


# tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index


# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)


# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)



# %% Text representation
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size= 50, window = 5, min_count= 1) # WARNING

embedding_dim = 50
embedding_mat = np.zeros((len(word_index) + 1, embedding_dim))


liste = []
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_mat[i] = word2vec_model.wv[word]
    else:
        liste.append(word)

# word embedding: word2vec


# %% Modeling

# build, train and test the RNN model
model = Sequential()

# embedding
model.add(Embedding(input_dim = len(word_index) + 1, output_dim = embedding_dim, weights = [embedding_mat], input_length = maxlen, trainable = False))

# RNN layer
model.add(SimpleRNN(50, return_sequences = False))

# Output layer
model.add(Dense(1, activation = "sigmoid"))


# compile model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# train model
model.fit(X_train, y_train, epochs = 10, batch_size = 2, validation_data = (X_test, y_test))


# evaluate RNN model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


# %% sentence classification study

def classify_sentence(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return label, prediction


sentence = "Restaurant harikaydı."

label, prediction = classify_sentence(sentence)

print(f"result: {label}")
print(f"predicted value: {prediction}")











