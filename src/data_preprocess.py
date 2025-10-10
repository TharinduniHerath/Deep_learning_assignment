# src/data_preprocess.py
import os, re, pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# parameters - tune these
MAX_VOCAB = 20000
MAX_LEN = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)                # remove HTML tags
    text = re.sub(r"[^a-zA-Z']", " ", text)           # keep words + apostrophes
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def main(csv_path="Data/IMDB Dataset.csv", out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    print("Raw shape:", df.shape)
    df['clean_review'] = df['review'].astype(str).apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['clean_review'].values
    y = df['label'].values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE,
                                                        stratify=y)

    # tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # save
    np.savez_compressed(os.path.join(out_dir, "data.npz"),
                        X_train=X_train_pad, X_test=X_test_pad, y_train=y_train, y_test=y_test)
    with open(os.path.join(out_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    print("Saved processed data to", out_dir)

if __name__ == "__main__":
    main()
