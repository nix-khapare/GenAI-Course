# Natural Language Processing (NLP) — Personal Knowledge Base

This README summarizes the NLP topics, techniques, hands-on experiments, and practical notes in this folder. It is written as a compact reference for future study, project work, or interview prep. Files referenced in this folder:

- `stemming.ipynb` — experiments with NLTK stemmers (Porter, RegexpStemmer, Snowball) and notes. 
- `limitizationipynb` — (present in folder) — likely about lemmatization/limiti(z)ation (review this notebook for more details).

## Quick contract

- Inputs: raw text data (sentences, documents)
- Outputs: cleaned/normalized text, features (TF-IDF, embeddings), model predictions (classification, NER, etc.)
- Typical error modes: noisy text, out-of-vocabulary words, domain mismatch, class imbalance
- Success criteria: correct preprocessing pipeline, reproducible experiments, evaluation metrics appropriate to task (accuracy/precision/recall/F1/AUROC)

## High-level concepts covered

- Text preprocessing: tokenization, lowercasing, stopword removal, punctuation removal, normalization
- Stemming vs lemmatization: aggressive suffix-stripping (stemming) vs linguistically informed normalization (lemmatization)
- Feature extraction: Bag-of-Words, n-grams, TF-IDF, word embeddings (Word2Vec, GloVe), contextual embeddings (BERT-style)
- Models and algorithms: Naive Bayes, Logistic Regression, SVM, simple neural nets, RNNs/LSTMs/GRUs, and Transformer-based models
- Common tasks: text classification (sentiment, spam), sequence labeling (POS tagging, NER), text similarity, information retrieval, summarization

## Tools & libraries used or recommended — details & quick interview Q/A

Below are the principal tools you’ve used or are recommended, with short descriptions and compact interview-style Q/A you can memorize and repeat during interviews.

- NLTK (Natural Language Toolkit)
	- What it is: A comprehensive educational library for classical NLP tasks: tokenization, stemming, lemmatization, corpora, and simple classifiers.
	- When to use: learning, prototyping classic NLP pipelines, and experimenting with algorithms (Porter stemmer, RegexpStemmer, WordNet lemmatizer).
	- Interview Q: "When would you choose NLTK vs spaCy?"
		- A: Use NLTK for learning and fine-grained algorithm access; use spaCy for production-ready, fast pipelines and robust tokenization/POS/NER.

- spaCy
	- What it is: Industrial-strength NLP library with fast tokenization, dependency parsing, POS tagging, named entity recognition, and pre-trained pipelines.
	- When to use: production, pipelines requiring speed and stability, and when you need easy model deployment.
	- Interview Q: "What are spaCy's strengths?"
		- A: Speed, good defaults, easy model serialization, and modern pre-trained pipelines for common languages.

- scikit-learn
	- What it is: Classic machine learning library (feature extraction, models, evaluation). It’s the go-to for TF-IDF, vectorizers, and simple classifiers like Logistic Regression and SVM.
	- Interview Q: "Why use TF-IDF + Logistic Regression instead of deep learning for a small text dataset?"
		- A: Simpler pipelines train faster, need less data, are easier to interpret and tune, and often match or beat deep models on small datasets.

- gensim
	- What it is: Tools for topic modeling (LDA) and training word embeddings like Word2Vec and FastText.
	- Interview Q: "When would you train embeddings with gensim vs use pre-trained embeddings?"
		- A: Train with gensim when domain-specific vocabulary matters and you have enough text; use pre-trained embeddings when data is limited and general semantic capture is needed.

- Hugging Face Transformers
	- What it is: State-of-the-art pre-trained transformer models (BERT, RoBERTa, DistilBERT) and a flexible training/finetuning ecosystem.
	- Interview Q: "What is transfer learning in NLP and why are Transformers useful for it?"
		- A: Transfer learning = starting from pre-trained language models and fine-tuning on a downstream task. Transformers capture deep contextual patterns and generalize well across tasks, reducing data needs.

- pandas / numpy
	- What they are: Data handling and numerical tools used to prepare datasets and perform evaluation.
	- Interview Q: "How do you prepare text data for modeling with pandas?"
		- A: Load data (CSV/JSON), clean text columns (remove NaNs, normalize), split into train/test, and convert text into features (TF-IDF or embeddings).

Quick tool-choice checklist (interview-ready):

- Need fast production pipeline -> spaCy
- Need classic features + interpretable model -> scikit-learn + TF-IDF
- Need contextual representation or SOTA -> Hugging Face Transformers
- Need simple experiments/teaching material -> NLTK

## Stemming: notes, practical details, and interview Q/A

Short definition and use-cases:

- What is stemming? A rule-based or algorithmic process that reduces words to their base or root form by chopping off affixes (e.g., "running" -> "run"). It is used to reduce vocabulary size for information retrieval and simple classification.
- When to use: IR tasks, rough normalization for bag-of-words models, or when exact morphological correctness is not required.

Common stemmers and practical notes:

- PorterStemmer
	- Pros: simple and widely used; good baseline.
	- Cons: aggressive; may over-truncate (produce non-words).

- SnowballStemmer
	- Pros: improved rules over Porter, language support, often more linguistically plausible stems.
	- Cons: still may produce stems that are not dictionary words.

- RegexpStemmer
	- Pros: highly controllable; useful for targeted suffix/prefix rules.
	- Cons: brittle for general text; requires careful regex crafting and minimum length checks.

Practical examples (conceptual outputs):

- Porter: 'eating' -> 'eat', 'congratulations' -> 'congratul'
- Snowball: 'congratulations' -> 'congratul' (often similar but sometimes more consistent)
- RegexpStemmer configured for 'ing$|s$|able$' and min length 4: 'eating' -> 'eat', 'disable' -> 'dis'

Interview Q/A (concise answers you can memorize):

- Q: "What is the difference between stemming and lemmatization?"
	- A: Stemming is a heuristic chopping of word endings (fast, may yield non-words). Lemmatization uses vocabulary and POS information to return dictionary lemmas (slower, more accurate).

- Q: "When would you prefer stemming over lemmatization?"
	- A: When speed and simplicity matter (e.g., large-scale IR), or when a rough normalization is sufficient and exact morphological correctness is not required.

- Q: "Give an example when stemming can hurt model performance."
	- A: In tasks where morphological form matters (e.g., intent detection where "meeting" vs "meet" changes meaning) or when downstream models rely on valid lemmas for embedding lookups.

- Q: "How do you choose between Porter, Snowball, and a regex stemmer?"
	- A: Start with Snowball for general use in English. Use Porter for a baseline. Use RegexpStemmer only for targeted rule-based normalization when you know the suffixes to strip.

- Q: "How would you evaluate whether stemming helps your pipeline?"
	- A: Run an ablation: compare model performance (accuracy/F1) on a validation set using raw text, stemmed text, and lemmatized text. Also inspect qualitative errors introduced by stemming.

- Q: "Can stemming be used together with embeddings?"
	- A: Usually no for pre-trained embeddings — stemming changes token forms so you lose the pre-trained vector mapping. For training embeddings from scratch, stemming is possible but often unnecessary; subword models (FastText, BPE) handle morphology better.

Practical tips and pitfalls:

- If you need real-word outputs (e.g., downstream generation or readable outputs), prefer lemmatization.
- If you use RegexpStemmer, include a minimum length parameter to avoid stripping too much (e.g., min=4).
- Always validate preprocessing choices with a quick controlled experiment; small datasets can flip which approach is best.

## Typical preprocessing pipeline

1. Text cleaning: remove HTML, normalize whitespace, correct encoding issues
2. Tokenization: word or subword tokenization (NLTK, spaCy, or tokenizer from Transformers)
3. Normalization: lowercase, remove punctuation, optionally expand contractions
4. Stopword removal (task dependent)
5. Stemming or lemmatization (pick one depending on downstream task)
6. Feature extraction: TF-IDF, n-grams, or embeddings
7. Model training / inference

Edge cases to watch for:

- Languages other than English — different tokenization rules and stemmers
- Domain-specific tokens (emails, URLs, code snippets) — treat them specially
- Handling emojis/Unicode and multilingual text

## Feature extraction & embeddings (study notes)

- Bag-of-Words / TF-IDF: simple, interpretable features for classical ML models. Works especially well with linear models for small datasets.
- Word embeddings (Word2Vec, GloVe, FastText): dense vector representations capturing semantic similarity. FastText helps with subword/OOV words.
- Contextual embeddings (BERT and friends): produce token/sequence-level vectors that capture context; excellent for transfer learning.

## Models & evaluation

- Common starter models: Multinomial Naive Bayes (text classification), Logistic Regression, SVMs
- Deep learning: simple feed-forward networks on TF-IDF features, RNNs/LSTMs for sequence modeling, Transformer architectures for state-of-the-art results
- Evaluation metrics:
	- Classification: accuracy, precision, recall, F1, confusion matrix, ROC-AUC (for binary tasks)
	- Sequence labeling: token-level F1, exact-match (for NER/QA)
	- Retrieval / ranking: MAP, nDCG

## Practical applications & project notes

- Experiments performed (in this folder):
	- Stemming experiments (`stemming.ipynb`): compared Porter, RegexpStemmer, Snowball Stemmer with examples and observations.

- Example projects to mention or expand:
	- Sentiment analysis: product review classification (binary or multi-class)
	- Spam detection: email or SMS classification (binary)
	- Named Entity Recognition (NER): extract person/organization/location from text using spaCy or a fine-tuned transformer
	- Text similarity / clustering: deduplicate documents or cluster similar items using embeddings + cosine similarity

## Small reproducible experiment (template)

1. Create a virtual environment and install dependencies (e.g., NLTK, scikit-learn, pandas, spaCy).
2. Run quick preprocessing on a sample dataset (CSV with text and label).
3. Convert to TF-IDF features and train a Logistic Regression classifier.
4. Evaluate with a hold-out test set and produce precision/recall/F1.

Minimal code sketch (conceptual):

	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import classification_report

	X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
	vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
	Xtr = vec.fit_transform(X_train)
	Xte = vec.transform(X_test)
	clf = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
	print(classification_report(y_test, clf.predict(Xte)))

## Interview-focused bullet list

- Explain the difference between stemming and lemmatization and when to use each.
- Describe TF-IDF and why it downweights common words.
- Given a small dataset (text + labels), outline preprocessing and a simple ML pipeline end-to-end.
- Describe pros/cons of pre-trained embeddings vs training embeddings from scratch.
- When to use classical ML (Naive Bayes/LogReg) vs deep learning (transformers).
