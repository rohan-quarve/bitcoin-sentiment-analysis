import re
import pandas as pd

IN_PATH = "data/raw/bitcoin_tweets.csv"
OUT_PATH = "data/processed/text_clean.csv"
SAMPLE_N = 200_000

def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"http\S+|www\.\S+", "", s)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"#", "", s)
    s = re.sub(r"&amp;|&lt;|&gt;", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    chunks = pd.read_csv(
        IN_PATH,
        usecols=["text"],     
        chunksize=200_000,
        on_bad_lines="skip"
    )

    cleaned = []
    for chunk in chunks:
        chunk["text_clean"] = chunk["text"].map(clean_text)
        chunk = chunk[chunk["text_clean"].str.len() >= 5]
        cleaned.append(chunk[["text_clean"]])
        if sum(len(c) for c in cleaned) >= SAMPLE_N:
            break

    df = pd.concat(cleaned, ignore_index=True).head(SAMPLE_N)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} cleaned documents")

def main():
    print("AI4SMA-II project initialized")

if __name__ == "__main__":
    main()
