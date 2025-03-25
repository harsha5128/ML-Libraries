import spacy

# 🔹 Load Pre-trained NLP Model
nlp = spacy.load("en_core_web_sm")

# 🔹 Sample Financial Text Data
text = """JP Morgan Chase reported a quarterly profit of $9.44 billion. 
Bitcoin surpassed $60,000 amid market fluctuations. 
The Federal Reserve announced an interest rate hike affecting global markets."""

# 🔹 Apply NER
doc = nlp(text)

# 🔹 Extract Named Entities
print("🔹 Extracted Entities:")
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")

# =====================================================
# 📌 Hyperparameters Explained
# =====================================================
"""
- `nlp = spacy.load("en_core_web_sm")`: Loads a small, efficient **English NLP model**.
- `doc.ents`: Extracts named entities (companies, monetary values, organizations).
"""

# =====================================================
# 📌 SUMMARY
# =====================================================
"""
✅ Why NER?
- Helps in **information extraction** from financial reports, news articles, and SEC filings.
- Identifies **companies, currencies, and market indicators**.

✅ Where is it used?
- **Automated risk assessment**: Extracts key financial terms from documents.
- **Market sentiment analysis**: Identifies companies and economic indicators.

✅ Advantages & Disadvantages:
| Advantages                         | Disadvantages                    |
|------------------------------------|---------------------------------|
| Helps automate document processing | May misclassify financial terms |
| Can be fine-tuned for FinTech use  | Requires training for custom data |

🔹 Best Practices:
- Use **custom financial models** (e.g., `FinBERT`) for better entity detection.
- Combine NER with **knowledge graphs** for deeper insights.
"""
