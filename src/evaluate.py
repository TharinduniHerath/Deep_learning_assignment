# src/evaluate.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_processed(path="artifacts/data.npz"):
    data = np.load(path)
    return data['X_test'], data['y_test']

def evaluate(model_path="artifacts/models/gru_best.h5"):
    X_test, y_test = load_processed()
    model = load_model(model_path)
    preds = model.predict(X_test, batch_size=256).ravel()
    pred_labels = (preds >= 0.5).astype(int)

    print(classification_report(y_test, pred_labels, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, preds))

    cm = confusion_matrix(y_test, pred_labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

if __name__ == "__main__":
    evaluate()
