

# Metinlerde bulunann boşlukları ortadan kaldır.
text = "Hello,    World!    2035"

cleaned_text1 = " ".join(text.split())
print(f"text: {text} \n cleaned_text1: {cleaned_text1} ")



# %% Büyük -> küçük harf çevrimi

text = "Hello, World! 2035"
cleaned_text2 = text.lower() # Lower the letters

print(f"text: {text} \n cleaned_text2: {cleaned_text2} ")




# %% Noktalama İşaretlerini kaldır

import string

text = "Hello, World! 2035"


cleaned_text3 = text.translate(str.maketrans("", "", string.punctuation))
print(f"text: {text} \n cleaned_text3: {cleaned_text3} ")



# %% Özel karakterleri kaldır %, @,/ * & $

import re

text = "Hello, World! 2035%"

cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]", "", text)

print(f"text: {text} \n cleaned_text4: {cleaned_text4} ")


# %% Yazım hatalarını düzelt

from textblob import TextBlob # for text analysis

text = "Hellio, Wirld! 2035"

cleaned_text5 = TextBlob(text).correct() # correct: punctuation

print(f"text: {text} \n cleaned_text5: {cleaned_text5} ")


 

# %% HTML ya da URL etiketlerini kaldır

from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2035</div>" # has a html label
cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
print(f"text: {html_text} \n cleaned_text6: {cleaned_text6} ")




























