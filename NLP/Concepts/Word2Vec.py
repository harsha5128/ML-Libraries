import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK tokenizers are downloaded
nltk.download('punkt')

# 📌 Sample corpus (Finance-related)
sentences = [
    "Stock market is experiencing high volatility",
    "Investors are concerned about inflation",
    "Interest rates impact economic growth",
    "Cryptocurrency is becoming popular among young investors",
    "Banks are implementing AI models for fraud detection"
]

# 🔹 Tokenizing sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# 🔹 Training Word2Vec Model (Skip-gram)
word2vec_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,  # 🔹 Number of dimensions for word vectors (higher captures more semantics)
    window=5,  # 🔹 Context window size (larger captures wider context)
    min_count=1,  # 🔹 Ignores words with occurrences below this threshold
    sg=1,  # 🔹 Skip-gram model (1=Skip-gram, 0=CBOW)
    workers=4,  # 🔹 Number of CPU cores for training
    epochs=10,  # 🔹 Number of training iterations
    hs=0,  # 🔹 Use Negative Sampling instead of Hierarchical Softmax
    negative=5  # 🔹 Number of negative samples (improves training)
)

# 🔹 Extracting vector for a word
word_vector = word2vec_model.wv["investors"]
print("🔹 Word Vector for 'investors':", word_vector[:5])  # Display first 5 elements

# 🔹 Finding similar words
similar_words = word2vec_model.wv.most_similar("market")
print("\n🔹 Words similar to 'market':", similar_words)

# =====================================================
# 📌 SUMMARY
# =====================================================
"""
✅ Why Word2Vec?
- Captures word semantics and relationships.
- Used in NLP tasks like sentiment analysis, topic modeling, and recommendation systems.

✅ Where is it used?
- In **FinTech**, Word2Vec helps in fraud detection, credit risk modeling, and automated trading strategies.

✅ Advantages & Disadvantages:
| Advantages                | Disadvantages                          |
|---------------------------|--------------------------------------|
| Captures word meaning    | Requires a large corpus              |
| Handles synonyms well    | Training can be slow                 |
| Good for semantic similarity | Doesn't handle out-of-vocabulary words |

🔹 Best Practices:
- Use **CBOW** for large datasets (faster, general training).
- Use **Skip-gram** for small datasets (better for rare words).
- Increase `vector_size` for complex applications but keep efficiency in mind.
"""
