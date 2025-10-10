# src/models.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_gru(max_vocab, embedding_dim=100, max_len=200, gru_units=128, dropout=0.5):
    model = models.Sequential([
        layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim, input_length=max_len),
        layers.SpatialDropout1D(0.2),
        layers.GRU(gru_units, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_lstm(max_vocab, embedding_dim=100, max_len=200, lstm_units=128, dropout=0.5):
    model = models.Sequential([
        layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim, input_length=max_len),
        layers.SpatialDropout1D(0.2),
        layers.LSTM(lstm_units),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_cnn(max_vocab, embedding_dim=100, max_len=200, dropout=0.5):
    model = models.Sequential([
        layers.Embedding(input_dim=max_vocab, output_dim=embedding_dim, input_length=max_len),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
