"""
Problem definition and dataset

Classify the comments in the amazon dataset as positive and negative

Binary classification problem

"""

# import libs
import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")



# load dataset
df = pd.read_csv("duygu_analizi_amazon_veri_seti.csv")


lemmatizer = WordNetLemmatizer()
# text cleaning and preprocessing
def clean_preprocess_data(text):
    # tokenize data
    tokens = word_tokenize(text.lower())
    
    # remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # find lemmas
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # join words
    processed_text = " ".join(lemmatized_tokens)
    
    return processed_text


df["reviewText2"] = df["reviewText"].apply(clean_preprocess_data)


# sentiment analysis (nltk)
analyzer = SentimentIntensityAnalyzer()


def get_sentiments(text):
    score = analyzer.polarity_scores(text)
    
    sentiment = 1 if score["pos"] > 0 else 0
    
    return sentiment
    

df["sentiment"] = df["reviewText2"].apply(get_sentiments)
    

# evaluation - test
from sklearn.metrics import confusion_matrix, classification_report


cm = confusion_matrix(df["Positive"], df["sentiment"])
print(f"Confusion Matrix: {cm} ")


"""
Confusion Matrix: [[ 1131  3636]
                  [  576 14657]] 
"""

cr = classification_report(df["Positive"],df["sentiment"]) 
print("Classification Report: \n{}".format(cr))



"""
Classification Report: 
              precision    recall  f1-score   support

           0       0.66      0.24      0.35      4767
           1       0.80      0.96      0.87     15233

    accuracy                           0.79     20000
   macro avg       0.73      0.60      0.61     20000
weighted avg       0.77      0.79      0.75     20000
"""








