# Sentiment Analysis using Naïve Bayes Classifier

Binary sentiment classification (positive / negative) on the IMDB Movie Reviews dataset using a Multinomial Naïve Bayes classifier with TF-IDF features.

---

## Dataset

**IMDB Movie Reviews** via `keras.datasets.imdb`

- **Source:** [Keras IMDB Dataset](https://keras.io/api/datasets/imdb/) — originally from Maas et al. (2011), *"Learning Word Vectors for Sentiment Analysis"*
- **Size:** 50,000 reviews total — 25,000 training + 25,000 test
- **Labels:** Binary — `0 = Negative`, `1 = Positive` (balanced)
- **Encoding:** Reviews are pre-encoded as integer sequences; each integer maps to a word ranked by corpus frequency
- **Vocabulary cap:** Top 10,000 most frequent words (`NUM_WORDS = 10000`)

---

## Project Pipeline

| Step | Description |
|------|-------------|
| 1. **Load Data** | Pull the IMDB dataset (25K train + 25K test) from `keras.datasets.imdb` |
| 2. **EDA** | Explore class balance, review length distributions, and most frequent words (bar chart, histogram, word cloud) |
| 3. **Decode** | Reverse the integer encoding back to plain-text reviews using the Keras word index |
| 4. **Preprocess** | Strip special tokens, lowercase, remove punctuation, remove stopwords, lemmatize (NLTK WordNet) |
| 5. **Feature Extraction** | Build a TF-IDF matrix (`max_features=20,000`, unigrams + bigrams, log-scaled TF, English stopwords removed) |
| 6. **Train** | Fit a Multinomial Naïve Bayes classifier (`alpha=0.1` Laplace smoothing) on the training features |
| 7. **Evaluate** | Predict on the test set; report accuracy, precision, recall, F1 (classification report + confusion matrix) |
| 8. **Sanity Check** | Predict sentiment on custom input sentences |
| 9. **Error Analysis** | Inspect high-confidence misclassifications to understand model limitations |

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **85.78%** |
| Precision (avg) | 0.86 |
| Recall (avg) | 0.86 |
| F1-score (avg) | 0.86 |

---

## Key Design Choices

- **TF-IDF with bigrams** — captures basic negation patterns (e.g. *"not good"*) that would be lost with unigrams alone.  Stopwords are removed *inside* the vectorizer so bigrams like *"not good"* are preserved before stopword filtering applies.
- **Sublinear TF scaling** — dampens the effect of very frequent words.
- **Laplace smoothing (α = 0.1)** — prevents zero-probability for words seen at test time but not in training.
- **Lemmatization (NLTK WordNet)** — reduces vocabulary size and groups morphological variants.

---

## Repository Structure

```
Assignment/
├── SentimentAnalysis_NaiveBayes.ipynb   # Main notebook
├── eda_charts.png                        # EDA visualisations (auto-generated)
├── confusion_matrix.png                  # Confusion matrix (auto-generated)
└── README.md                             # This file
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `tensorflow` / `keras` | IMDB dataset loader |
| `scikit-learn` | TF-IDF vectorizer, Multinomial NB, evaluation metrics |
| `nltk` | WordNet lemmatizer |
| `wordcloud` | Word cloud visualisation |
| `numpy`, `pandas` | Data handling |
| `matplotlib`, `seaborn` | Plotting |

### Install optional dependencies

```bash
pip install nltk wordcloud
```

NLTK WordNet data is downloaded automatically inside the notebook:

```python
nltk.download('wordnet', quiet=True)
```

---

## How to Run

1. Clone / download this repository.
2. Open `SentimentAnalysis_NaiveBayes.ipynb` in Jupyter Lab / Notebook.
3. Run **all cells in order** (the first cell contains commented-out `pip install` commands for `nltk` and `wordcloud` — uncomment if needed).
4. The IMDB dataset is downloaded automatically on first run (~17 MB).

---

## References

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL 2011.
- [Keras IMDB Dataset Documentation](https://keras.io/api/datasets/imdb/)
- [Scikit-learn Multinomial Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
