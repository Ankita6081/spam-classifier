"""
train.py - Train the Spam Classifier model
Run this first to train and save the model.
Usage: python train.py
"""

import os
import pickle
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── Preprocessing ────────────────────────────────────────────────────────────

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Clean, tokenise, remove stopwords, and stem a message."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)          # keep only letters
    tokens = text.split()
    tokens = [ps.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# ── Load Dataset ─────────────────────────────────────────────────────────────

DATA_PATH = os.path.join('data', 'spam.csv')

print(f"\nLoading dataset from {DATA_PATH} ...")
try:
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
except FileNotFoundError:
    print(f"\n[ERROR] Dataset not found at '{DATA_PATH}'.")
    print("Please download 'spam.csv' from:")
    print("  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print("and place it inside the 'data/' folder, then re-run this script.")
    raise SystemExit(1)

# The UCI CSV has columns v1 (label) and v2 (message); drop unnamed extras
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

print(f"Dataset loaded: {len(df)} messages  |  "
      f"Spam: {df['label'].sum()}  |  Ham: {(df['label']==0).sum()}")

# ── Preprocess ───────────────────────────────────────────────────────────────

print("\nPreprocessing messages...")
df['clean'] = df['message'].apply(preprocess)

# ── Train / Test Split ───────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ── Vectorise ────────────────────────────────────────────────────────────────

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Train Model ───────────────────────────────────────────────────────────────

print("Training Multinomial Naive Bayes classifier...")
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_vec)

print("\n" + "="*50)
print("         MODEL EVALUATION RESULTS")
print("="*50)
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Ham  (TN): {cm[0][0]}   False Spam (FP): {cm[0][1]}")
print(f"  False Ham (FN): {cm[1][0]}   True Spam  (TP): {cm[1][1]}")
print("="*50)

# ── Save Artefacts ────────────────────────────────────────────────────────────

os.makedirs('model', exist_ok=True)
with open(os.path.join('model', 'model.pkl'),      'wb') as f:
    pickle.dump(model, f)
with open(os.path.join('model', 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved to the 'model/' directory.")
print("You can now run:  python app.py")
