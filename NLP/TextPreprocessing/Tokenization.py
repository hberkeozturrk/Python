
import nltk # Natural Language Toolkit



nltk.download("punkt_tab") # Tokenize the text in terms of words and sentences

# text = "Hello, world! How are you? Hello, hi..."

text = """Artificial intelligence (AI) is the simulation of human intelligence by machines,  
especially computer systems. It enables machines to learn from data, adapt to new  
inputs, and perform tasks traditionally requiring human intelligence, such as  
speech recognition, decision-making, and visual perception. AI is widely used  
across industries, from healthcare and finance to transportation and entertainment.  
Its applications include virtual assistants, recommendation systems, and autonomous  
vehicles. As AI continues to evolve, it presents both opportunities and challenges—  
enhancing productivity and innovation, while also raising ethical concerns about  
privacy, job displacement, and accountability. Its future depends on responsible  
development and thoughtful integration into society."""




# Word tokenization: word_tokenize
word_tokens = nltk.word_tokenize(text)


# Sentence tokenization: sentence_tokenize
sentence_tokens = nltk.sent_tokenize(text)







