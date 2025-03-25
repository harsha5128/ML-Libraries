import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔹 Load Pre-trained GloVe Model (50D)
glove_model = api.load("glove-wiki-gigaword-50")  # 50-dimensional word embeddings

# 🔹 Fetch vector for a word
word_vector = glove_model["finance"]
print("🔹 GloVe Vector for 'finance':", word_vector[:5])  # Display first 5 elements

# 🔹 Find similar words
similar_words = glove_model.most_similar("investment", topn=5)
print("\n🔹 Words similar to 'investment':", similar_words)

# 🔹 Compute Cosine Similarity between two words
vector1 = glove_model["loan"]
vector2 = glove_model["credit"]
similarity_score = cosine_similarity([vector1], [vector2])
print(f"\n🔹 Cosine Similarity between 'loan' and 'credit': {similarity_score[0][0]:.4f}")

# =====================================================
# 📌 SUMMARY
# =====================================================
"""
✅ Why GloVe?
- **Combines benefits of Word2Vec & traditional co-occurrence matrices.**
- Captures **word relationships using matrix factorization**.
- Works well for **semantic similarity tasks**.

✅ Where is it used?
- In **FinTech**, GloVe helps analyze financial news sentiment, risk assessment, and fraud detection.

✅ Advantages & Disadvantages:
| Advantages                | Disadvantages                          |
|---------------------------|--------------------------------------|
| Pre-trained models available | Doesn't capture context dynamically |
| Efficient & accurate    | Requires significant computational resources |
| Captures global word co-occurrence | Not good for polysemy (multiple meanings) |

🔹 Best Practices:
- Use pre-trained **GloVe embeddings** instead of training from scratch.
- Fine-tune embeddings for **domain-specific applications** (like finance).
"""
