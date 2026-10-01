import pandas as pd
import numpy as np
import re
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from tfidf import tfidf_fit_transform, tfidf_transform
from logisticRegression import (
    split_data,
    train_model,
    evaluate_model
)

#setting up dataset and paramaters
DATA_PATH = r"/Users/rohanquarve/school/sp2026/VIP/AI4SMA-II/tweets.csv"
MIN_DF = 10
LEARNING_RATE = 1.0
NUM_ITERS = 300


#cleaning the tweets
def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"http\S+|www\.\S+", "", s)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"#", "", s)
    s = re.sub(r"&amp;|&lt;|&gt;", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

#lstm help
def create_sequences(data, seq_len=30):
    X, y = [], []

    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 1])  # predicting percent change

    return np.array(X), np.array(y)

#confusion matrix plot
def plot_confusion(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

#precision and recall graph
def plot_precision_recall(model, X, y):
    probs = model.predict_probabilities(X)
    precision, recall, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={ap:.3f})")
    plt.show()

def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Loss Over Epochs")
    plt.legend()
    plt.show()

def plot_accuracy(train_accs, val_accs):
    plt.figure()
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Over Epochs")
    plt.legend()
    plt.show()



#full pipeline
def main():

    parser = argparse.ArgumentParser(description="Bitcoin Tweet Logistic Regression Pipeline")
    parser.add_argument("--max_rows", type=int, default=50000,
                        help="Maximum number of rows to load from dataset")
    parser.add_argument("--save_model", action="store_true",
                        help="Save trained model to disk")
    args = parser.parse_args()

    #load the dataset
    print("Loading dataset...")
    df = pd.read_csv(
        DATA_PATH,
        usecols=["text", "movement", "date"],
        nrows=args.max_rows
    )

    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    print(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=["text", "movement"])

    #cleaning
    print("Cleaning text...")
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() >= 5]

    #encode labels
    print("Encoding labels...")
    df["label"] = df["movement"].str.lower().map({
        "increase": 1,
        "decrease": 0
    })

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    texts = df["text_clean"].values
    y = df["label"].values

    print(f"Dataset size after cleaning: {len(texts):,}")

    #split data
    print("Splitting data...")
    X_train_texts, y_train, X_val_texts, y_val, X_test_texts, y_test = split_data(
        texts, y
    )

    #tf-idf
    print("Fitting TF-IDF on training set...")
    X_train, vocab, idf = tfidf_fit_transform(
        X_train_texts,
        min_df=MIN_DF
    )

    print("Transforming validation set...")
    X_val = tfidf_transform(X_val_texts, vocab, idf)

    print("Transforming test set...")
    X_test = tfidf_transform(X_test_texts, vocab, idf)

    print(f"Vocabulary size: {len(vocab):,}")

    #training model
    print("Training Logistic Regression...")
    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=LEARNING_RATE,
        num_iters=NUM_ITERS
    )

    #evaluate model
    print("\n--- Validation Set ---")
    evaluate_model(model, X_val, y_val, dataset_name="Validation")

    print("\n--- Test Set ---")
    evaluate_model(model, X_test, y_test, dataset_name="Test")  

    #plotting graphs
    print("\nPlotting Confusion Matrix (Test Set)...")
    plot_confusion(model, X_test, y_test)

    print("\nPlotting Precision-Recall Curve (Test Set)...")
    plot_precision_recall(model, X_test, y_test)

    print("\nPlotting Loss Curve...")
    plot_loss(model.train_losses, model.val_losses)
    plt.show()

    print("\nPlotting Accuracy Curve...")
    plot_accuracy(model.train_accuracies, model.val_accuracies) 
    plt.show()

    # aggregating data needed for lstm
    BTC_PATH = "bitcoin_2017_to_2023.csv"

    print("Loading Bitcoin dataset...")

    btc = pd.read_csv(BTC_PATH)

    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc["date"] = btc["timestamp"].dt.date

    btc = btc.sort_values("timestamp")

    btc_daily = btc.groupby("date").last().reset_index()

    btc_daily["daily_percent_change"] = btc_daily["close"].pct_change() * 100

    btc_subset = btc_daily[["date", "close", "daily_percent_change"]]


    # generate tweet sentiment
    print("Generating sentiment scores for all tweets...")

    X_all = tfidf_transform(texts, vocab, idf)

    sentiment_probs = model.predict_probabilities(X_all)

    df["sentiment"] = sentiment_probs


    # aggregate sentiment per day
    daily_sentiment = df.groupby("date")["sentiment"].mean().reset_index()


    # merge sentiment with BTC data
    lstm_data = btc_subset.merge(daily_sentiment, on="date", how="left")
    lstm_data = lstm_data.fillna(0)

    features = lstm_data[["close", "daily_percent_change", "sentiment"]].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    # prepare LSTM input

    X_lstm, y_lstm = create_sequences(features, seq_len=7)

    # split LSTM data
    split = int(0.8 * len(X_lstm))

    X_train_lstm = X_lstm[:split]
    X_test_lstm = X_lstm[split:]

    y_train_lstm = y_lstm[:split]
    y_test_lstm = y_lstm[split:]
    
    # train LSTM
    print("\nTraining LSTM model...")

    model_lstm = Sequential()
    model_lstm.add(LSTM(32, input_shape=(7,3)))
    model_lstm.add(Dense(16, activation="relu"))
    model_lstm.add(Dense(1))

    model_lstm.compile(optimizer="adam", loss="mse")

    model_lstm.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=10,
    batch_size=16
    )
    print("\nEvaluating LSTM model...")

    preds = model_lstm.predict(X_test_lstm)

    mse = mean_squared_error(y_test_lstm, preds)
    mae = mean_absolute_error(y_test_lstm, preds)

    print(f"LSTM Mean Squared Error: {mse:.4f}")
    print(f"LSTM Mean Absolute Error: {mae:.4f}")
    
    plt.figure()
    plt.plot(y_test_lstm, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("LSTM Prediction vs Actual")
    plt.show()
    
    print("\nPipeline complete.")
    
if __name__ == "__main__":
    main()
