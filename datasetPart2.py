import pandas as pd

# --- Load tweet dataset ---
tweets = pd.read_csv("/kaggle/working/bitcoin_subset.csv")

# Convert created_at to datetime and extract date
tweets["created_at"] = pd.to_datetime(tweets["created_at"])
tweets["date"] = tweets["created_at"].dt.date


# --- Load Bitcoin price dataset ---
btc = pd.read_csv("/kaggle/input/datasets/jkraak/bitcoin-price-dataset/bitcoin_2017_to_2023.csv")

# Convert timestamp column to datetime
btc["timestamp"] = pd.to_datetime(btc["timestamp"])

# Extract just the date
btc["date"] = btc["timestamp"].dt.date

btc = btc.sort_values("timestamp")
btc_daily = btc.groupby("date").last().reset_index()

# Calculate daily percent change using daily close
btc_daily["daily_percent_change"] = btc_daily["close"].pct_change() * 100

# Keep only needed columns
btc_subset = btc_daily[["date", "close", "daily_percent_change"]]

# --- Merge tweets with BTC price data ---
merged = tweets.merge(btc_subset, on="date", how="left")

# make sure there are enough english tweets
english_count = (merged["lang"] == "en").sum()
print(f"Number of English tweets in final dataset: {english_count}")

# Create increase/decrease column
merged["movement"] = np.where(
    merged["daily_percent_change"] > 0,
    "increase",
    np.where(
        merged["daily_percent_change"] < 0,
        "decrease",
        "no_change"
    )
)

# Save final dataset
merged.to_csv("/kaggle/working/tweets_with_btc.csv", index=False)

print("Final dataset shape:", merged.shape)
print(merged.head())


# --- Distribution of returns ---
pos_count = (merged['daily_percent_change'] > 0).sum()
neg_count = (merged['daily_percent_change'] < 0).sum()
zero_count = (merged['daily_percent_change'] == 0).sum()

print("\nReturn Distribution for Tweet Dates:")
print(f"Positive returns: {pos_count}")
print(f"Negative returns: {neg_count}")
print(f"No change (0%): {zero_count}")

merged.head()
