# Bitcoin Sentiment & Price Prediction

A machine learning pipeline for analyzing Bitcoin-related tweets and investigating the relationship between social-media sentiment and Bitcoin price movement.

The project combines **natural language processing (NLP)** with **machine learning and time-series modeling** to:

* Clean and preprocess Bitcoin-related tweets
* Convert tweet text into numerical features using TF-IDF
* Predict daily Bitcoin price direction using logistic regression
* Generate tweet-level and daily aggregated sentiment scores
* Combine sentiment with historical Bitcoin price data
* Use an LSTM to model short-term Bitcoin price movement

## Project Overview

The pipeline consists of two primary modeling stages.

### 1. Tweet Classification

Bitcoin tweets are cleaned and labeled according to the Bitcoin market movement associated with their date:

* `increase` → `1`
* `decrease` → `0`

The cleaned tweets are transformed into TF-IDF features and used to train a custom logistic regression classifier.

The classifier is implemented from scratch using NumPy and gradient descent, with training and validation loss and accuracy tracked throughout training.

### 2. Price Prediction with Sentiment

After training the classifier, its predicted probabilities are used as a **sentiment score** for tweets.

The scores are aggregated by day and combined with:

* Bitcoin closing price
* Daily percentage price change
* Daily tweet sentiment

These features are then passed into an LSTM using seven-day sequences to predict the next day's Bitcoin percentage change.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── bitcoin_tweets.csv
│   └── processed/
│       └── text_clean.csv
│
├── src/
│   ├── datasetPart1.py
│   ├── datasetPart2.py
│   ├── TF_IDF.py
│   ├── logisticRegression.py
│   ├── LSTM.py
│   └── main.py
│
└── README.md
```

### Source Files

| File                    | Purpose                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `datasetPart1.py`       | Samples and prepares the original Bitcoin tweet dataset                                   |
| `datasetPart2.py`       | Combines tweet data with historical Bitcoin price data and creates market-movement labels |
| `TF_IDF.py`             | Implements TF-IDF feature extraction from scratch                                         |
| `logisticRegression.py` | Implements logistic regression, dataset splitting, training, and evaluation               |
| `LSTM.py`               | Contains LSTM-related functionality and sequence preparation                              |
| `main.py`               | Runs the complete NLP → classification → sentiment → LSTM pipeline                        |

---

## Data Pipeline

```text
Bitcoin Tweets
      │
      ▼
Data Cleaning
      │
      ▼
TF-IDF Feature Extraction
      │
      ▼
Logistic Regression
      │
      ├──────────────► Classification Metrics
      │
      ▼
Tweet-Level Sentiment Scores
      │
      ▼
Daily Sentiment Aggregation
      │
      ├──────────────┐
      │              │
      ▼              ▼
Bitcoin Price    Daily % Change
      │              │
      └───────┬──────┘
              ▼
        Combined Features
              │
              ▼
       7-Day LSTM Sequences
              │
              ▼
       Bitcoin Price Prediction
```

## Dataset Preparation

The initial dataset preparation script loads Bitcoin tweets, converts timestamps to datetime values, filters for English-language tweets, and samples the dataset.

The second preparation stage combines the tweets with historical Bitcoin price data. Daily closing prices and percentage changes are calculated, and each tweet is associated with an `increase`, `decrease`, or `no_change` market-movement label.

## Text Preprocessing

Tweets are normalized before modeling:

* Convert text to lowercase
* Remove URLs
* Remove user mentions
* Remove hashtag symbols
* Remove common HTML entities
* Remove non-alphanumeric characters
* Normalize whitespace

Tweets shorter than five characters after cleaning are discarded.

## TF-IDF

The project implements TF-IDF without relying on a pre-built TF-IDF vectorizer.

For each training dataset, the implementation:

1. Tokenizes documents
2. Builds a vocabulary
3. Applies a minimum document-frequency threshold
4. Calculates inverse document frequency
5. Constructs the TF-IDF feature matrix

The same vocabulary and IDF values are reused when transforming validation and test data, preventing the feature representation from changing between datasets.
The main pipeline currently uses:

```text
Minimum document frequency: 10
```

## Logistic Regression

A custom logistic regression implementation is used instead of scikit-learn's classifier.

The model uses:

* Sigmoid activation
* Log-loss
* Gradient descent
* Configurable learning rate
* Configurable number of iterations
* Training and validation accuracy tracking

The default training configuration in `main.py` is:

```text
Learning rate: 1.0
Iterations:    300
```

The model evaluates performance using accuracy and a classification report.

### Evaluation

The pipeline generates:

* Validation accuracy
* Test accuracy
* Classification report
* Confusion matrix
* Precision-recall curve
* Training loss curve
* Validation loss curve
* Training accuracy curve
* Validation accuracy curve

These visualizations are generated by the main pipeline after model training.

## Sentiment Generation

Once logistic regression is trained, its predicted probabilities are calculated for all cleaned tweets.

These probabilities are used as a continuous sentiment score:

```text
Tweet → TF-IDF → Logistic Regression → Probability → Sentiment Score
```

The scores are then averaged for each date to produce a daily sentiment feature.

## LSTM Model

The LSTM combines three daily features:

```text
1. Bitcoin closing price
2. Daily percentage change
3. Tweet-derived sentiment
```

The features are normalized using `MinMaxScaler` and converted into seven-day sequences.

The current LSTM architecture is:

```text
LSTM(32)
    ↓
Dense(16, ReLU)
    ↓
Dense(1)
```

The model uses:

* Adam optimizer
* Mean squared error loss
* 10 epochs
* Batch size of 16
* 7-day input sequences

The final model is evaluated using:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

An actual-versus-predicted plot is also generated.

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Python 3.10+ is recommended.

## Running the Project

The main entry point is:

```text
src/main.py
```

Run the pipeline with:

```bash
python src/main.py
```

You can optionally limit the number of tweet records loaded:

```bash
python src/main.py --max_rows 50000
```

The `--max_rows` argument controls how many rows are loaded from the tweet dataset.

---

## Configuration

Several parameters can be modified in `src/main.py`:

```python
MIN_DF = 10
LEARNING_RATE = 1.0
NUM_ITERS = 300
```

The LSTM configuration is currently defined in the training section:

```text
Sequence length: 7 days
LSTM units:      32
Dense units:     16
Epochs:          10
Batch size:      16
```

---

## Expected Data

The pipeline expects tweet data containing at least:

```text
text
date
movement
```

The historical Bitcoin dataset should contain:

```text
timestamp
close
```

The data-preparation pipeline creates daily Bitcoin percentage changes from the closing prices.

### Important

The current scripts contain dataset paths that were used in the original development environment, including Kaggle and local filesystem paths. These paths should be updated to match your local repository before running the project.

## For example, `main.py` currently defines the tweet dataset path directly in the source code, while the LSTM portion expects `bitcoin_2017_to_2023.csv`.

## Technologies

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **scikit-learn**
* **TensorFlow / Keras**
* **Natural Language Processing**
* **TF-IDF**
* **Logistic Regression**
* **LSTM Neural Networks**

## Key Concepts

This project explores:

* Text preprocessing
* Feature engineering
* TF-IDF representation
* Binary classification
* Gradient-descent optimization
* Model validation
* Precision and recall
* Sentiment scoring
* Time-series feature engineering
* Sequential neural networks
* Bitcoin price prediction

## Limitations

The project should be viewed as an experimental machine-learning study rather than a financial forecasting system.

In particular:

* Tweet sentiment is derived from a classifier trained on market-movement labels.
* The LSTM uses historical market data and tweet-derived sentiment rather than a broader set of market indicators.
* The current pipeline uses a seven-day sequence for the LSTM.
* Dataset paths are currently environment-specific.
* The relationship between social-media sentiment and Bitcoin price movement does not imply causation.

## Future Improvements

Potential extensions include:

* Use a chronological rather than randomized split for the classification stage
* Experiment with transformer-based language models such as BERT
* Add additional market features such as trading volume and volatility
* Tune the LSTM architecture and sequence length
* Compare LSTM predictions against simpler time-series baselines
* Perform more rigorous time-series validation
* Add model checkpointing and experiment tracking
* Move dataset paths into configuration files or command-line arguments
* Add automated tests for preprocessing and model components

---

## Authors

**Rohan Quarve**, **Jared Dong**, **Harshini Madhusundhanan**, **Max Kirschner**

Developed as a machine-learning project exploring the relationship between Bitcoin-related social-media activity and cryptocurrency market movement.


## Contributions

This project was developed collaboratively as part of a larger team effort, with different team members contributing to different components of the machine-learning pipeline.

My contributions included:

* **TF-IDF:** Helped develop and implement the custom TF-IDF text-processing pipeline used to convert tweet text into numerical features.
* **LSTM:** Independently designed and implemented the LSTM-based time-series modeling component, including:

  * Combining daily Bitcoin price data with aggregated tweet-derived sentiment
  * Engineering the features used by the time-series model
  * Normalizing features with `MinMaxScaler`
  * Creating seven-day sequential input windows
  * Designing and training the LSTM model
  * Evaluating the model using MSE and MAE
  * Visualizing predicted versus actual Bitcoin prices

The LSTM component was my primary individual contribution, while the remaining parts of the project were developed collaboratively with other team members.
