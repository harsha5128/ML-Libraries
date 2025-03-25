import numpy as np
import pandas as pd
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# 🔹 Sample Financial Text Data
documents = [
    "Stock markets are highly volatile and require risk analysis.",
    "Loan default prediction helps in assessing credit risk.",
    "Cryptocurrency is gaining popularity in financial transactions.",
    "Banking fraud detection uses machine learning and AI models.",
    "Insurance companies evaluate customer risk using predictive analytics.",
    "FinTech innovations are transforming payment systems.",
    "Algorithmic trading automates stock market investments."
]

# 🔹 Tokenization (Required for LDA)
nltk.download("punkt")
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# 🔹 Create Dictionary & Corpus
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

# 🔹 Train LDA Model
lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# 🔹 Display Topics
topics = lda_model.print_topics(num_words=5)
print("🔹 Identified Topics:")
for topic in topics:
    print(topic)

# =====================================================
# 📌 Hyperparameters Explained
# =====================================================
"""
- `num_topics=3`: **Number of topics to extract**.
- `passes=10`: **Number of iterations over the corpus to refine topics**.
- `id2word=dictionary`: **Mapping of words to their token IDs**.
"""

# =====================================================
# 📌 SUMMARY
# =====================================================
"""
✅ Why LDA?
- It helps in **automatic topic discovery** from financial documents.
- Useful for **fraud detection**, risk assessment, and FinTech applications.

✅ Where is it used?
- **Regulatory compliance**: Extracting key topics from large reports.
- **Risk modeling**: Understanding customer sentiment from financial texts.

✅ Advantages & Disadvantages:
| Advantages                          | Disadvantages                     |
|-------------------------------------|----------------------------------|
| Useful for discovering hidden themes | Requires pre-processing |
| Works well for large datasets       | Choosing the number of topics is tricky |

🔹 Best Practices:
- Use **TF-IDF transformation** before LDA for better topic separation.
- Tune **num_topics** carefully to avoid underfitting/overfitting.
"""
