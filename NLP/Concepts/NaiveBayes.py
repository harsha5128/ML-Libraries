# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------
# 🔹 Sample Text Dataset (Spam vs Ham)
# -------------------
data = {
    "text": [
        "Congratulations! You have won a free lottery ticket. Call now!",
        "Urgent! Your account has been compromised. Reset your password immediately!",
        "Hello friend, how are you doing today?",
        "Win a brand new iPhone by clicking this link.",
        "Meeting at 5 PM. Let me know if you can join.",
        "Your loan application has been approved. Click here to claim."
    ],
    "label": [1, 1, 0, 1, 0, 1]  # 1 = Spam, 0 = Ham (Normal Message)
}

df = pd.DataFrame(data)

# -------------------
# 🔹 Text Preprocessing (TF-IDF Vectorization)
# -------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------
# 🔹 Train Naïve Bayes Model
# -------------------
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Naïve Bayes Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------
# 🔹 Hyperparameter Tuning (Grid Search)
# -------------------
param_grid = {
    "alpha": [0.1, 0.5, 1.0]  # Laplace smoothing parameter
}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Parameters & Model Performance
best_nb = grid_search.best_estimator_
best_params = grid_search.best_params_
print("\n🔹 Best Hyperparameters:", best_params)

# Evaluate Best Model
y_pred_best = best_nb.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"\nNaïve Bayes Accuracy (Tuned): {best_accuracy:.4f}")
