

from transformers import pipeline

import warnings

warnings.filterwarnings("ignore")



# summarize pipeline
summarizer = pipeline("summarization")



text = """
Machine learning (ML) is a branch of artificial intelligence that
focuses on enabling systems to learn from data and improve their
performance without being explicitly programmed. It involves
creating algorithms that can identify patterns, make decisions, and
adapt based on experience. ML is widely used across various
industries, including healthcare, finance, transportation, and
e-commerce, due to its ability to automate and enhance
decision-making processes.

There are three main types of machine learning: supervised
learning, unsupervised learning, and reinforcement learning.
Supervised learning relies on labeled datasets to train models to
make predictions or classify data. Unsupervised learning deals with
unlabeled data and aims to discover hidden structures or patterns
within it. Reinforcement learning involves an agent that learns to
make decisions by interacting with an environment and receiving
feedback through rewards or penalties.

The success of machine learning heavily depends on the quality and
quantity of the data provided. As data becomes more abundant and
accessible, ML continues to advance, powering innovations such as
recommendation systems, natural language processing, image
recognition, and autonomous vehicles. Despite its strengths,
machine learning also raises concerns around ethics, bias, and
data privacy, making responsible development and use increasingly
important.
"""


# text summarization

summary = summarizer(
    text,
    max_length = 20,
    min_length = 5,
    do_sample = True
    )
        

print(summary[0]["summary_text"])


"""
Machine learning (ML) is a branch of artificial intelligence that
enables systems to learn from data and improve their performance
without being explicitly programmed . ML is widely used across various industries
including healthcare, finance, transportation, and e-commerce . 
As data becomes more abundant and accessible, ML continues to advance .

"""


"""
 Machine learning (ML) is a branch of artificial intelligence that enables systems to learn from

"""










