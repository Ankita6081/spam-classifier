# SMS / Email Spam Classifier

A machine learning project that classifies SMS and email messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and a Multinomial Naive Bayes classifier.

## Problem Statement

Spam messages are a persistent daily nuisance — from fake prize notifications to phishing attempts. This project builds a text classification pipeline that automatically detects whether an incoming message is spam, using only the content of the message itself.

## How It Works

1. **Preprocessing** — Messages are lowercased, stripped of punctuation, tokenised, stopwords are removed, and words are reduced to their root form using Porter Stemming.
2. **Feature Extraction** — TF-IDF vectorisation (unigrams + bigrams) converts text into numerical features.
3. **Classification** — A Multinomial Naive Bayes model (well-suited for text data) is trained on the SMS Spam Collection dataset.

## Project Structure

```
spam-classifier/
├── data/
│   └── spam.csv          # Dataset (see setup instructions)
├── model/
│   ├── model.pkl         # Saved model (generated after training)
│   └── vectorizer.pkl    # Saved vectorizer (generated after training)
├── train.py              # Trains and saves the model
├── predict.py            # Core prediction logic (importable module)
├── app.py                # Command-line interface
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Ankita6081/spam-classifier.git
cd spam-classifier
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the **SMS Spam Collection Dataset** from Kaggle:

👉 https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

- Download `spam.csv` from the page
- Place it inside the `data/` folder:

```
spam-classifier/
└── data/
    └── spam.csv    ← place it here
```

### 5. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train the Naive Bayes classifier
- Print accuracy and evaluation metrics
- Save the model to `model/model.pkl` and `model/vectorizer.pkl`

**Expected output:**
```
Accuracy: ~98%
```

### 6. Run the App

**Interactive mode** (type your own messages):
```bash
python app.py
```

**Demo mode** (runs on built-in sample messages):
```bash
python app.py --demo
```

**Single message mode:**
```bash
python app.py --msg "Congratulations! You've won a free iPhone. Claim now!"
```

## Example Output

```
─────────────────────────────────────────────────────────────
  Message   : Congratulations! You've won a free iPhone. Claim now!
  Verdict   : 🚨  SPAM  (98.7% confidence)
  Spam prob :  98.7%  [██████████████████████████████]  Ham: 1.3%
─────────────────────────────────────────────────────────────

─────────────────────────────────────────────────────────────
  Message   : Are we still meeting at 5pm today?
  Verdict   : ✅  HAM   (99.2% confidence)
  Spam prob :   0.8%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  Ham: 99.2%
─────────────────────────────────────────────────────────────
```

## Dataset

**SMS Spam Collection Dataset** — UCI Machine Learning Repository  
5,574 English SMS messages labelled as `ham` (legitimate) or `spam`.  
Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~98%   |
| Precision | ~99%   |
| Recall    | ~94%   |
| F1-Score  | ~96%   |

## Course Concepts Applied

- Text preprocessing (tokenisation, stemming, stopword removal)
- TF-IDF feature extraction
- Naive Bayes classification
- Train/test split and model evaluation
- Precision, Recall, F1-Score, Confusion Matrix
- Model serialisation with `pickle`

## Author

Ankita Sharma  
Reg. No.: 25BAS10041  
VIT Bhopal University
