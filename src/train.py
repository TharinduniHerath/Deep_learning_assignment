# src/train.py
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.models import build_gru
import argparse
import json

def load_data(path="artifacts/data.npz"):
    data = np.load(path)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def train(model_name="gru", epochs=8, batch_size=128, out_dir="artifacts/models"):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    max_vocab = 20000  # must match preprocessing Tokenizer.num_words
    max_len = X_train.shape[1]

    if model_name == "gru":
        model = build_gru(max_vocab, embedding_dim=100, max_len=max_len)
    else:
        raise ValueError("Model not supported in this script (extend as needed)")

    model.compile(optimizer=Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    ckpt_path = os.path.join(out_dir, f"{model_name}_best.h5")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]

    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks)

    # save final model and training history
    model.save(os.path.join(out_dir, f"{model_name}_final.keras"))

    with open(os.path.join(out_dir, f"{model_name}_history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gru")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train(model_name=args.model, epochs=args.epochs)
