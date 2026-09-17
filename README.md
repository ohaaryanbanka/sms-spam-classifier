# SMS Spam Classifier

A machine learning project that classifies text messages as **spam** or **ham** (not spam) using Multinomial Naive Bayes, with support for both English and Hinglish (Hindi-English mixed) messages.

Built as a college project for **CS (AI & ML)** coursework at PES University.

## Results

Final tuned model (alpha = 0.5, decision threshold = 0.55):

- **Accuracy:** 98.53%
- **Spam precision:** 98.25%
- **Spam recall:** 89.6%
- **Spam F1-score:** 93.72%

![Confusion Matrix](final_confusion_matrix.png)

## How it works

1. **Dataset** — the public "SMS Spam Collection" dataset (5,572 real SMS messages) combined with an additional 1,000-message sample, then deduplicated. Each message is labeled `ham` or `spam`.
2. **Text cleaning** — lowercased, URLs and emails stripped, punctuation removed, then stemmed (e.g. "winning"/"wins" → "win") using `PorterStemmer`.
3. **Stopword removal** — both English filler words and a custom Hinglish stopword list (hai, ka, ki, mein, kya, etc.) are filtered out, so the model works across both languages.
4. **Vectorization** — `CountVectorizer` with unigrams and bigrams (`ngram_range=(1,2)`), ignoring very rare terms (`min_df=2`) and overly common ones (`max_df=0.95`).
5. **Model** — `MultinomialNB`, with the smoothing parameter (`alpha`) tuned across a range of values and selected by spam F1-score.
6. **Threshold tuning** — instead of the default 50% cutoff, the decision threshold for calling a message "spam" was tuned separately (0.20–0.60 range) and set to 0.55, the point of best spam F1-score.
7. **Evaluation** — accuracy, precision/recall/F1, confusion matrix, and manual review of missed spam messages, all computed on a held-out test set.

## Project structure

| File | Description |
|---|---|
| `spamandham.ipynb` | Full notebook — cleaning, vectorization, alpha tuning, threshold tuning, evaluation, error analysis |
| `spam_classifier.py` | Earlier baseline script (CountVectorizer + Naive Bayes, no tuning) |
| `spam_gui.py` | Desktop GUI version built with Tkinter — type a message, get an instant spam/ham verdict |
| `sms_spam_dataset.tsv` | Original dataset (tab-separated: `label`, `message`) |
| `sms_spam_dataset_1000.tsv` | Additional 1,000-message sample merged into training data |
| `final_confusion_matrix.png` | Confusion matrix for the final tuned model |

## How to run it

### Notebook (Google Colab / Jupyter)
1. Upload `sms_spam_dataset.tsv` and `sms_spam_dataset_1000.tsv` to your session.
2. Open `spamandham.ipynb` and run all cells top to bottom (**Runtime → Restart session → Runtime → Run all** recommended for a clean run).

### Desktop app (Tkinter)
Requires Python 3 with `pandas` and `scikit-learn` installed locally:
```bash
pip install pandas scikit-learn
python spam_gui.py
```
Make sure `sms_spam_dataset.tsv` is in the same folder as `spam_gui.py`.

## Tech stack

- Python 3
- pandas — data loading and handling
- scikit-learn — vectorization, model training, evaluation
- nltk — stemming (`PorterStemmer`)
- numpy — threshold sweep
- matplotlib + seaborn — confusion matrix visualization
- Tkinter — desktop GUI (standard library, no install needed)

## License

MIT — see [LICENSE](LICENSE).
