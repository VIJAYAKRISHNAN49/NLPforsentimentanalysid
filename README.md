# Customer Review Sentiment Analysis using NLP and LSTM

## Project Overview and Objective
Organizations receive large volumes of customer reviews that are impractical to analyze
manually. This project builds a deep learning pipeline that classifies movie reviews (as a
stand-in for customer reviews) as **Positive** or **Negative**, covering the full path from raw
text to a trained **LSTM** classifier: text cleaning, tokenization, traditional feature
representations (Bag of Words, TF-IDF), word embeddings, and an LSTM-based sentiment model.

## Dataset Details
- **Dataset:** IMDB Movie Reviews, loaded via `tensorflow.keras.datasets.imdb` (downloads
  automatically — no manual download required).
- **Size:** 50,000 reviews total — 25,000 training, 25,000 testing, perfectly balanced between
  classes.
- **Labels:** 0 = Negative, 1 = Positive.
- **Format:** Keras provides this dataset **pre-tokenized as integers**, with `num_words=10000`
  keeping only the 10,000 most frequent words (rarer words are dropped). A word-index decoder is
  used in this notebook to reconstruct raw text for the manual preprocessing steps below.

## Preprocessing Steps
1. **Decode to raw text** — Keras's word index is inverted to reconstruct human-readable text
   from the integer-encoded reviews, so the classic NLP pipeline can be demonstrated on real words.
2. **Text cleaning** — lowercasing, punctuation/number removal.
3. **Tokenization** — splitting cleaned text into individual word tokens.
4. **Stopword removal** — filtering common low-signal words ("the", "is", "and", ...) via NLTK.
5. **Stemming and lemmatization** — both demonstrated (Porter Stemmer and WordNet Lemmatizer) as
   alternative ways to normalize words to a root form.
6. **Word indexing** — Keras's `num_words=10000` vocabulary cutoff *is* the word-indexing step
   for the data actually fed to the model.
7. **Sequence generation and padding** — `pad_sequences` makes every review exactly 200 tokens
   long (padding shorter reviews, truncating longer ones).
8. **Train/validation/test split** — the provided 25k/25k train/test split, with a further 10%
   of the training set held out for validation during training.

## Text Representation — Bag of Words vs. TF-IDF vs. Word Embeddings
| | Bag of Words | TF-IDF | Word Embeddings |
|---|---|---|---|
| Vector type | Sparse, raw counts | Sparse, importance-weighted | Dense, learned |
| Dimensionality | = vocabulary size (large) | = vocabulary size (large) | Small, fixed (128 here) |
| Captures word order | No | No | Yes (when fed into an RNN/LSTM) |
| Captures word meaning/similarity | No | No | Yes |
| Used by | Traditional ML (Logistic Regression, Naive Bayes, SVM) | Traditional ML | Deep learning (RNN, LSTM, Transformers) |

A **Bag of Words** and **TF-IDF** representation are both built (via scikit-learn) on a subset of
decoded reviews for demonstration, alongside a quick TF-IDF + Logistic Regression baseline
classifier — giving a traditional-ML comparison point against the LSTM's deep-learning approach.
**Word embeddings** are used directly inside the LSTM model itself (the `Embedding` layer),
learned end-to-end during training rather than computed as a separate preprocessing step.

## Model Architecture
```
Input (200,) padded integer sequence
 -> Embedding(vocab_size=10000, embedding_dim=128)
 -> LSTM(64, dropout=0.2, recurrent_dropout=0.2)
 -> Dense(32, ReLU) -> Dropout(0.3)
 -> Dense(1, Sigmoid)
```
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Callbacks:** `EarlyStopping` (patience=3, on validation loss) and `ModelCheckpoint` (saves
  the best validation-accuracy checkpoint)

## Project Structure
```
NLP_Sentiment_LSTM/
│
├── dataset/                                # (auto-downloaded by Keras on first run)
├── nlp_sentiment_analysis_lstm.ipynb       # Full implementation notebook (Tasks 1-5)
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation (this file)
└── output_images/                          # Confusion matrix, training curves
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook nlp_sentiment_analysis_lstm.ipynb
   ```
3. Run all cells top to bottom. The IMDB dataset and NLTK resources (stopwords, punkt, wordnet)
   download automatically on first run.
4. Outputs (confusion matrix, training curves) are saved to `output_images/`, and the trained
   model is saved as `lstm_sentiment_model.h5`.

> A GPU runtime (Colab: Runtime → Change runtime type → GPU) will speed up training, though this
> LSTM is small enough to also train in a reasonable time on CPU.

## Training Results
*(Update this section with your actual numbers after running the notebook.)*

| Metric | Value |
|---|---|
| Test Accuracy | TBD |
| Test Precision | TBD |
| Test Recall | TBD |
| Test F1 Score | TBD |
| TF-IDF + Logistic Regression baseline (2000-review subset) | TBD |

## Evaluation Metrics
The notebook reports accuracy, precision, recall, F1 score, a full classification report, a
confusion matrix heatmap, training-vs-validation accuracy/loss curves, and a side-by-side table
of actual vs. predicted sentiment on sample test reviews (with model confidence).

## Performance Analysis and Conclusion

**Impact of sequence length:** reviews shorter than `MAX_LEN` (200) are zero-padded; longer ones
are truncated, discarding content beyond the window. Too short a length loses sentiment-bearing
text; too long wastes computation on mostly-padding sequences.

**Importance of the embedding layer:** converts arbitrary word IDs (categorical, not ordinal)
into dense vectors the network can meaningfully learn from — without it, the model would
incorrectly treat word IDs as if they had numeric magnitude/order.

**Effect of LSTM memory cells:** the gated cell state lets the model retain context across long
stretches of text, which matters for reviews that pivot mid-way (e.g. "great acting, but a
terrible plot") — something a bag-of-words approach structurally cannot capture, since it ignores
word order entirely.

**Overfitting and solutions:** watch for a widening gap between training and validation
accuracy/loss. Mitigations already applied: dropout/recurrent_dropout inside the LSTM, an
additional Dropout layer, and EarlyStopping. Further options: fewer LSTM units, L2
regularization, more training data, or pretrained embeddings (e.g. GloVe).

**Limitations:** vocabulary cutoff at 10,000 words drops rarer terms; no explicit handling of
sarcasm/irony; standard (unidirectional) LSTM only reads left-to-right (a Bidirectional LSTM
could improve this); binary-only output can't express degrees of sentiment or neutral reviews;
a model trained on movie reviews may not transfer well to other review domains.

**Conclusion:** update this with your specific findings (final metrics, whether overfitting was
observed, how the LSTM compared to the TF-IDF baseline) once you've run the notebook.

## Author
Data Science Module 5 — Assignment 11: NLP Model – Sentiment Analysis using LSTM
