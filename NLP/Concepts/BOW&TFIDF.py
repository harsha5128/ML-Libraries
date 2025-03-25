# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ------------------------------------------
# 🔹 Sample Text Data
# ------------------------------------------
corpus = [
    "NLP is an exciting field of AI.",
    "Machine Learning enables NLP advancements.",
    "Deep Learning improves NLP models.",
    "Text processing is crucial in NLP.",
    "NLP is widely used in chatbots."
]

# ------------------------------------------
# 🔹 Bag-of-Words (BoW) Representation
# ------------------------------------------
# BoW converts text into a word frequency matrix, where each row represents a document 
# and each column represents a word from the vocabulary.
vectorizer = CountVectorizer()  # Initialize BoW vectorizer
X_bow = vectorizer.fit_transform(corpus)  # Convert text corpus into a sparse matrix

# Display BoW results
print("\n🔹 BoW Vocabulary:", vectorizer.get_feature_names())  # List of words in vocabulary
print("\n🔹 BoW Matrix:\n", X_bow.toarray())  # Word frequency matrix

# ------------------------------------------
# 🔹 TF-IDF (Term Frequency-Inverse Document Frequency)
# ------------------------------------------
# TF-IDF improves BoW by reducing the importance of common words and highlighting important terms.
tfidf_vectorizer = TfidfVectorizer()  # Initialize TF-IDF vectorizer
X_tfidf = tfidf_vectorizer.fit_transform(corpus)  # Convert text corpus into TF-IDF representation

# Display TF-IDF results
print("\n🔹 TF-IDF Vocabulary:", tfidf_vectorizer.get_feature_names())  # List of words in vocabulary
print("\n🔹 TF-IDF Matrix:\n", X_tfidf.toarray())  # TF-IDF matrix with normalized importance scores

# ------------------------------------------
# 🔹 Summary Notes
# ------------------------------------------
# ✅ Why Use BoW & TF-IDF?
# - BoW represents text based on word occurrence (word frequency).
# - TF-IDF improves BoW by reducing the importance of common words.

# 🔹 Where & When to Use?
# - Feature extraction in NLP models.
# - Text classification & clustering.
# - Search engines (TF-IDF for ranking results).

# 🔻 Advantages:
# ✔ Simple and effective text representation.
# ✔ Works well with traditional ML models.

# 🔻 Disadvantages:
# ✖ Ignores word order & context.
# ✖ Not suitable for capturing semantic meaning.
