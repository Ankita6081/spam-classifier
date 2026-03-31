"""
predict.py - Core prediction module
Loads the saved model and exposes a predict() function used by app.py.
"""

import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

MODEL_PATH      = os.path.join('model', 'model.pkl')
VECTORIZER_PATH = os.path.join('model', 'vectorizer.pkl')

ps         = PorterStemmer()
stop_words = set(stopwords.words('english'))


def _load_artifacts():
    """Load model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model files not found. Please run 'python train.py' first."
        )
    with open(MODEL_PATH,      'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def _preprocess(text: str) -> str:
    """Apply the same preprocessing used during training."""
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)


def predict(message: str) -> dict:
    """
    Classify a single SMS / email message.

    Parameters
    ----------
    message : str
        The raw message text.

    Returns
    -------
    dict with keys:
        label       – 'SPAM' or 'HAM'
        confidence  – float, probability of the predicted class (0–1)
        spam_prob   – float, probability the message is spam (0–1)
        ham_prob    – float, probability the message is ham  (0–1)
    """
    model, vectorizer = _load_artifacts()

    cleaned  = _preprocess(message)
    vec      = vectorizer.transform([cleaned])
    proba    = model.predict_proba(vec)[0]   # [P(ham), P(spam)]

    ham_prob  = proba[0]
    spam_prob = proba[1]
    label     = 'SPAM' if spam_prob > ham_prob else 'HAM'
    confidence = spam_prob if label == 'SPAM' else ham_prob

    return {
        'label':      label,
        'confidence': round(float(confidence)  * 100, 2),
        'spam_prob':  round(float(spam_prob)   * 100, 2),
        'ham_prob':   round(float(ham_prob)    * 100, 2),
    }


if __name__ == '__main__':
    # Quick sanity-check when run directly
    samples = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still meeting at 5pm today?",
        "URGENT: Your bank account has been compromised. Call 0800-XXX-XXXX immediately.",
        "Can you pick up some milk on your way home?",
    ]
    for msg in samples:
        result = predict(msg)
        print(f"[{result['label']}] ({result['confidence']}%) — {msg[:60]}")
