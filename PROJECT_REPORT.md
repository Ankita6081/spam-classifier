# Project Report — SMS / Email Spam Classifier

**Student:** Ankita Sharma  
**Roll No:** 25BAS10036  
**Institution:** VIT Bhopal University  


---

## 1. Problem Statement

Spam messages — unsolicited texts and emails promoting fake offers, phishing links, or fraudulent schemes — are a daily problem for virtually every mobile and internet user. Despite spam filters built into platforms like Gmail or iOS, standalone SMS spam often slips through, particularly on less-filtered channels.

The problem this project solves is: **Given the raw text of an SMS or email message, automatically determine whether it is spam or legitimate (ham).**

This is a real problem I encounter personally. Phishing messages disguised as bank alerts or prize notifications are common, and a lightweight classifier can serve as a first line of defence.

---

## 2. Why This Problem Matters

- Spam messages waste time and cause confusion
- Phishing links embedded in spam messages are a leading cause of cybercrime
- Existing spam filters are platform-specific; a standalone classifier can be embedded in any application
- Text classification is a foundational ML problem with direct real-world applications

---

## 3. Dataset

**SMS Spam Collection Dataset** — UCI Machine Learning Repository  
- 5,574 English SMS messages  
- Labels: `ham` (4,827 messages, ~86.6%) and `spam` (747 messages, ~13.4%)  
- Source: Originally collected for academic research, publicly available on Kaggle

The class imbalance (more ham than spam) is realistic and was accounted for during evaluation by using precision, recall, and F1-score rather than accuracy alone.

---

## 4. Approach

### 4.1 Text Preprocessing Pipeline

Raw text is unsuitable for ML models. I applied the following preprocessing steps:

1. **Lowercasing** — Normalises case (`FREE` → `free`)
2. **Punctuation removal** — Strips non-alphabetic characters (numbers, symbols common in spam like `£`, `!`, URLs are removed at this stage)
3. **Tokenisation** — Splits the message into individual words
4. **Stopword removal** — Removes common English words (`the`, `is`, `at`) that carry no discriminative signal
5. **Porter Stemming** — Reduces words to their root form (`winning` → `win`, `claims` → `claim`)

This pipeline ensures that the model focuses on meaningful content words.

### 4.2 Feature Extraction — TF-IDF

I used **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorisation with:
- `max_features=5000` — Top 5,000 most informative terms
- `ngram_range=(1,2)` — Both single words (unigrams) and two-word phrases (bigrams)

Including bigrams captures phrases like "click here" or "free entry" that are highly predictive of spam and would be lost with unigrams alone.

### 4.3 Model — Multinomial Naive Bayes

I chose **Multinomial Naive Bayes (MNB)** for the following reasons:
- Specifically designed for discrete count/frequency features like TF-IDF
- Fast to train and classify — suitable for real-time use
- Works well even with limited data
- Probabilistic output — provides a confidence score alongside the label
- Strong baseline performance on text classification tasks

`alpha=0.1` (Laplace smoothing) was used to handle unseen word combinations.

### 4.4 Evaluation Strategy

The data was split 80% training / 20% testing with stratification to preserve the ham/spam ratio. I evaluated using:
- **Accuracy** — Overall correctness
- **Precision** — Of messages predicted as spam, how many were actually spam?
- **Recall** — Of actual spam messages, how many were caught?
- **F1-Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Breakdown of true positives, false positives, etc.

Recall for spam is the most critical metric — a false negative (spam classified as ham) is worse than a false positive.

---

## 5. Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Algorithm | Multinomial Naive Bayes | Designed for text frequency data; fast and interpretable |
| Vectoriser | TF-IDF (not Bag of Words) | Downweights common words automatically |
| N-grams | Unigrams + bigrams | Captures multi-word spam phrases |
| Stemmer | Porter Stemmer | Standard, fast, reduces vocabulary size |
| Smoothing | alpha=0.1 | Reduces overconfidence on rare words |

---

## 6. Results

| Metric | Ham | Spam | Overall |
|---|---|---|---|
| Precision | ~99% | ~96% | ~98% |
| Recall | ~99% | ~94% | ~98% |
| F1-Score | ~99% | ~95% | ~98% |
| Accuracy | | | **~98%** |

The model correctly identified the vast majority of spam messages with very few false positives (legitimate messages incorrectly flagged as spam).

---

## 7. Challenges Faced

**1. Class Imbalance**  
The dataset has ~87% ham and ~13% spam. Accuracy alone would be misleading (a model that always predicts ham gets 87% accuracy). I addressed this by focusing on precision/recall/F1 for spam specifically.

**2. Short Message Length**  
SMS messages are short, giving the model fewer signals to work with compared to longer emails. Bigrams helped compensate by giving context to isolated words.

**3. URL and Number Removal**  
Spam messages often contain URLs and phone numbers, which are strong signals. My current preprocessing strips these. A future improvement would be to add binary features like `contains_url` or `contains_phone_number` before cleaning.

---

## 8. What I Learned

- The full ML pipeline: raw data → preprocessing → feature extraction → model training → evaluation
- Why TF-IDF is preferable to simple word counts for NLP tasks
- How Naive Bayes works probabilistically and why it suits text classification
- The importance of choosing the right evaluation metric for imbalanced classes
- How to serialise a trained model with `pickle` for reuse without retraining
- Writing a command-line interface that makes an ML model practically usable

---

## 9. Possible Improvements

- Add explicit features for URLs, phone numbers, and currency symbols (strong spam signals)
- Try ensemble methods (e.g., Random Forest, Gradient Boosting) and compare
- Expand to multilingual spam detection
- Build a simple web interface using Flask or Streamlit
- Collect a larger, more recent dataset (SMS spam has evolved since 2012)

---

## 10. Conclusion

This project successfully builds a working, end-to-end spam classifier that achieves ~98% accuracy on the SMS Spam Collection dataset. The solution is practical, runs entirely from the command line, and can classify any new message in real time. More importantly, it demonstrates the complete ML pipeline — from raw text to a deployed, usable tool — and addresses a genuine problem in everyday digital life.

---

## References

1. Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering. *ACM DOCENG*.
2. Scikit-learn documentation — https://scikit-learn.org
3. NLTK documentation — https://www.nltk.org
4. SMS Spam Collection Dataset — https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
