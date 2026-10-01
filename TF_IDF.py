import math
import numpy as np
from collections import Counter

def tfidf_fit_transform(texts, min_df=1):
    # returns np array
    docs = [str(t).split() for t in texts]
    N = len(docs)

    df = Counter()
    for d in docs:
        df.update(set(d))

    words = [w for w, c in df.items() if c >= min_df]
    vocab = {w: i for i, w in enumerate(words)}

    V = len(vocab)
    idf = np.zeros(V, dtype=np.float32)
    for w, j in vocab.items():
        idf[j] = math.log((N + 1) / (df[w] + 1)) + 1.0

    X = np.zeros((N, V), dtype=np.float32)
    for i, d in enumerate(docs):
        if not d:
            continue
        cnt = Counter(d)
        total = len(d)
        for w, c in cnt.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = (c / total) * idf[j]

    return X, vocab, idf


def tfidf_transform(texts, vocab, idf):
    # returns an NP array using existing vocab + idf
    docs = [str(t).split() for t in texts]
    N = len(docs)
    V = len(vocab)

    X = np.zeros((N, V), dtype=np.float32)
    for i, d in enumerate(docs):
        if not d:
            continue
        cnt = Counter(d)
        total = len(d)
        for w, c in cnt.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = (c / total) * float(idf[j])

    return X
