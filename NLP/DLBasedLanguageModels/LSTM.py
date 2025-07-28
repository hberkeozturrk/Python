
"""
Text generation
lstm train with text data
text data = gpt ile olustur
"""


# import libs
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




# create dataset with chatgpt

texts = [
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Kahvemi alıp balkonda oturmak harika bir his.",
    "Sabah erken kalkmak başta zor ama günümü güzelleştiriyor.",
    "Yeni müzikler keşfetmek ruh halimi olumlu etkiliyor.",
    "Yağmurun sesi beni hep huzurlu hissettiriyor.",
    "Gün batımını izlemek en sevdiğim anlardan biri.",
    "Arkadaşlarımla vakit geçirmek moralimi yükseltiyor.",
    "Temiz ve düzenli bir ortamda çalışmak daha verimli olmamı sağlıyor.",
    "Bazen sessizlik en iyi terapi olabiliyor.",
    "Güzel bir film izlemek ruh halimi değiştiriyor.",
    "Sabah kahvaltısı yapmadan güne başlayamıyorum.",
    "Doğada yürüyüş yapmak zihnimi dinlendiriyor.",
    "Kedimle vakit geçirmek beni rahatlatıyor.",
    "Müzik dinlemek beni bambaşka yerlere götürüyor.",
    "Bir fincan çay eşliğinde kitap okumak gibisi yok.",
    "Pencereden gelen güneş ışığı içimi ısıtıyor.",
    "Yeni bir şeyler öğrenmek beni heyecanlandırıyor.",
    "Resim yapmak beni özgür hissettiriyor.",
    "Yalnız kalmak bazen çok iyi geliyor.",
    "Parkta çocukların oynamasını izlemek keyif veriyor.",
    "Bir arkadaşla uzun uzun sohbet etmek iyi geliyor.",
    "Yeni yerler keşfetmek bana ilham veriyor.",
    "Sabah spor yapmak günümü enerjik başlatıyor.",
    "Sessiz bir ortamda çalışmak daha kolay oluyor.",
    "Sevdiğim bir yemeği yemek modumu düzeltiyor.",
    "Günlük tutmak duygularımı anlamama yardımcı oluyor.",
    "Sıcak bir duş almak günün yorgunluğunu alıyor.",
    "Evi temizledikten sonra oluşan ferahlık hissi güzel.",
    "Yıldızları izlemek beni huzurlu yapıyor.",
    "Yeni tarifler denemek mutfağı keyifli hale getiriyor.",
    "Bir hedefe ulaşmak insana tatmin duygusu veriyor.",
    "Aileyle geçirilen zamanın kıymeti çok büyük.",
    "Deniz kenarında oturmak iç huzur veriyor.",
    "Hayvanlarla vakit geçirmek stres azaltıyor.",
    "Kahkahalarla gülmek ruhuma iyi geliyor.",
    "Rüzgarın yüzüme çarpması hoş bir his.",
    "Yeni bir deftere yazmaya başlamak heyecan verici.",
    "Sabah kahvesi içmeden güne başlayamıyorum.",
    "Bir şeyleri tamamlamak motivasyonumu artırıyor.",
    "Güzel anılarımı düşünmek mutlu ediyor.",
    "Gün sonunda rahatlamak bana iyi geliyor.",
    "Sıcak battaniye altında film izlemek harika.",
    "Sevdiğim insanlarla olmak içimi ısıtıyor.",
    "Kütüphane sessizliği huzur veriyor.",
    "Manzara fotoğrafı çekmek beni heyecanlandırıyor.",
    "Bir dilek tutup yıldız kaymasını izlemek umut veriyor.",
    "Mektup yazmak nostaljik ama anlamlı bir deneyim.",
    "Düzenli plan yapmak beni güvende hissettiriyor.",
    "Bitki yetiştirmek sabır ve huzur getiriyor.",
    "Gönüllü çalışmalara katılmak anlamlı hissettiriyor.",
    "Sabah güneşinin doğuşunu izlemek bambaşka bir his.",
    "Favori şarkımı dinlemek her şeyi güzelleştiriyor.",
    "Tertemiz çarşaflarda uyumak müthiş rahatlatıcı.",
    "Pencereyi açıp temiz hava almak iyi geliyor.",
    "Sevdiğim bir diziyi izlemek bana keyif veriyor.",
    "Hediyeler hazırlamak ve vermek mutlu ediyor.",
    "Kendime zaman ayırmak bana çok iyi geliyor.",
    "Yağmur sonrası toprak kokusunu seviyorum.",
    "Çiçeklerin açışını izlemek umut veriyor.",
    "Evcil hayvanımla oyun oynamak eğlenceli.",
    "Yeni bir dil öğrenmeye başlamak heyecan veriyor.",
    "Sahilde yürümek beni rahatlatıyor.",
    "Bir projeyi tamamlamak içimi ferahlatıyor.",
    "Müzik eşliğinde temizlik yapmak daha keyifli.",
    "Yalnız başıma bir kafede oturmak bana iyi geliyor.",
    "Zamanı verimli kullanmak tatmin edici.",
    "Güneşli günlerde enerji doluyorum.",
    "Yeni kitaplar almak beni mutlu ediyor.",
    "Uzun yürüyüşler düşüncelerimi toparlıyor.",
    "Doğayla baş başa kalmak zihnimi açıyor.",
    "Kış günlerinde sıcak içecekler içmek çok keyifli.",
    "Kalabalıktan uzak zaman geçirmek bana huzur veriyor.",
    "İnsanlara yardım etmek kendimi iyi hissettiriyor.",
    "Pozitif sözler duymak günümü güzelleştiriyor.",
    "Güzel bir manzaraya bakmak içimi açıyor.",
    "Yolculuk yaparken müzik dinlemek ayrı bir keyif.",
    "Eski fotoğraflara bakmak beni duygulandırıyor.",
    "Bir işi başarıyla tamamlamak gurur verici.",
    "Hoş kokan bir parfüm moralimi düzeltiyor.",
    "Yeni bir hobi edinmek hayatı renklendiriyor.",
    "Etrafı düzenlemek zihnimi de sadeleştiriyor.",
    "Sıcak bir çorba içmek içimi ısıtıyor.",
    "Kendi başıma sinemaya gitmek özgür hissettiriyor.",
    "Kamp yapmak doğayla bütünleşmemi sağlıyor.",
    "Bisiklet sürmek hem spor hem eğlence.",
    "Taze meyve yemek sağlıklı hissettiriyor.",
    "Motivasyon konuşmaları dinlemek beni etkiliyor.",
    "Not almak ve yazmak beni organize ediyor.",
    "Hikaye yazmak duygularımı dışa vuruyor.",
    "Göl kenarında oturmak huzur veriyor.",
    "Sevdiğim şiirleri yeniden okumak keyifli.",
    "Evi mumlarla aydınlatmak farklı bir atmosfer yaratıyor.",
    "Kendime sessiz zaman tanımak bana iyi geliyor.",
    "Sade ve doğal olmak iç huzur veriyor.",
    "Yeni bir podcast keşfetmek heyecanlandırıyor.",
    "Ruh halime göre müzik listesi oluşturmak hoşuma gidiyor.",
    "Güne esneme hareketleriyle başlamak canlandırıcı.",
    "Gün içinde küçük molalar vermek çok işe yarıyor.",
    "Kendime küçük ödüller vermek motive ediyor.",
]




# %% Text cleaning and preprocessing: tokenization, padding, label encoding

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) # describe the frequencies on texts 
total_words = len(tokenizer.word_index) + 1 # total word count


# create n-gram arrays and apply padding
input_sequences = []

for text in texts:
    # convert texts into idx lists
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # create n-gram arrs for each text
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# padding (find the longest array), then make each array at same size
max_sequence_length = max(len(x) for x in input_sequences)

# Apply padding to the arrays
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding = "pre")

# X(input) and y(target) 
X = input_sequences[::-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes = total_words) # one-hot encoding




# %% Create LSTM model

# compile
model = Sequential()

# embedding
model.add(Embedding(total_words, 50, input_length = X.shape[1]))


# LSTM
model.add(LSTM(100, return_sequences = False))


# output
model.add(Dense(total_words, activation = "softmax"))


# compile 
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# model training
model.fit(X, y, epochs = 100, verbose = 1)


# %% Model prediction

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        # convert input text into numeric data
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # padding
        token_list = pad_sequences([token_list], maxlen = max_sequence_length - 1, padding = "pre")
        
        # prediction
        predicted_probs = model.predict(token_list, verbose = 0)
        
        # find the idx whose prob is the most
        predicted_word_index = np.argmax(predicted_probs, axis = -1)

        # find the actual word using the tokenizer
        predicted_word = tokenizer.index_word[predicted_word_index[0]]

        # add the guessed word to seed_text
        seed_text = seed_text + " " +  predicted_word
        
    return seed_text


seed_text = "Evden"

print(generate_text(seed_text, 4))








