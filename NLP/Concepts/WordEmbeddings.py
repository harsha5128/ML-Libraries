import gensim
from gensim.models import Word2Vec

# 🔹 Sample Financial Sentences
sentences = [
    ["JP", "Morgan", "Chase", "reported", "quarterly", "profit"],
    ["Bitcoin", "surpassed", "60000", "amid", "market", "fluctuations"],
    ["Federal", "Reserve", "announced", "interest", "rate", "hike"]
]

# 🔹 Train a Word2Vec Model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 🔹 Get Word Embeddings
print("🔹 Word Vector for 'Bitcoin':\n", model.wv["Bitcoin"])

# 🔹 Find Similar Words
print("\n🔹 Similar Words to 'market':\n", model.wv.most_similar("market"))

# =====================================================
# 📌 Hyperparameters Explained
# =====================================================
"""
- `vector_size=100`: Each word is represented as a **100-dimensional vector**.
- `window=5`: Looks at **5 words before and after** a target word for context.
- `min_count=1`: Includes words appearing at least **once** in the corpus.
- `workers=4`: Uses **4 CPU cores** for parallel training.
"""

# =====================================================
# 📌 SUMMARY
# =====================================================
"""
✅ Why Word Embeddings?
- Captures **semantic meaning** of financial terms.
- Helps in **market sentiment analysis & fraud detection**.

✅ Where is it used?
- **Stock price prediction**: Analyzes past financial news.
- **Customer risk profiling**: Identifies risky behavior in financial transactions.

✅ Advantages & Disadvantages:
| Advantages                         | Disadvantages                      |
|------------------------------------|-----------------------------------|
| Preserves word relationships       | Needs a **large dataset**         |
| Captures semantic meaning          | Doesn't handle **out-of-vocab words** |

🔹 Best Practices:
- Use **pre-trained embeddings** like **GloVe or FastText** for better financial term recognition.
- Fine-tune on **financial documents** (SEC filings, earnings reports) for better accuracy.
"""
