import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample DNA dataset
sequences = [
    "ATGCGTAC", "CGTACGTA", "TATGTGCA", "GCGTATGC",
    "TTATGGCA", "ATGCATGC", "GGCATGCA", "TATATATA",
    "CGCGATAT", "ATATCGCG", "GGGCCCGA", "TTTAAACC"
]

labels = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]

# Convert DNA sequences into k-mers (k=3)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
X = vectorizer.fit_transform(sequences)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

# -------- Logistic Regression --------
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# -------- Support Vector Machine --------
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\n=== Support Vector Machine ===")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
