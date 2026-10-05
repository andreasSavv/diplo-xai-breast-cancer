"""
autoencoder.py

Autoencoder-based classifier (2 στάδια), όπως στο Άρθρο 4 (XAI-MethylMarker):

Στάδιο Α (Autoencoder): ένα δίκτυο μαθαίνει να συμπιέζει τους 9 βιοδείκτες
σε μια μικρότερη αναπαράσταση (bottleneck) και μετά να τους "ξαναφτιάχνει".
Χρησιμοποιούμε MLPRegressor του sklearn με 1 κρυφό επίπεδο - αυτό το κρυφό
επίπεδο ΕΙΝΑΙ το bottleneck.

Στάδιο Β (Classifier): παίρνουμε τη συμπιεσμένη αναπαράσταση (όχι τα
αρχικά 9 νούμερα) και εκπαιδεύουμε πάνω της έναν απλό classifier.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def relu(x):
    return np.maximum(0, x)


def encode(autoencoder, X):
    """Υπολογίζει το bottleneck (συμπιεσμένη αναπαράσταση) για δεδομένα X,
    κάνοντας forward pass μόνο μέχρι το κρυφό επίπεδο (δεν χρειαζόμαστε την
    ανακατασκευή, μόνο τη συμπίεση)."""
    W1 = autoencoder.coefs_[0]       # βάρη: είσοδος -> bottleneck
    b1 = autoencoder.intercepts_[0]  # bias του bottleneck
    return relu(np.asarray(X) @ W1 + b1)


BOTTLENECK_SIZES = [3, 5, 7]

if __name__ == "__main__":
    from data_loading import load_wdbc_dataset

    data = load_wdbc_dataset()

    best_result = None
    print("Δοκιμή διαφορετικού μεγέθους bottleneck (συμπιεσμένης αναπαράστασης):\n")

    for bottleneck in BOTTLENECK_SIZES:
        # --- Στάδιο Α: εκπαίδευση autoencoder (μαθαίνει να αναπαράγει την είσοδο) ---
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(bottleneck,),
            activation="relu",
            solver="adam",
            max_iter=3000,
            random_state=42,
        )
        autoencoder.fit(data["X_train"], data["X_train"])  # στόχος = η ίδια η είσοδος

        reconstruction = autoencoder.predict(data["X_test"])
        recon_error = mean_squared_error(data["X_test"], reconstruction)

        # --- Στάδιο Β: εξαγωγή bottleneck + εκπαίδευση classifier πάνω του ---
        X_train_encoded = encode(autoencoder, data["X_train"])
        X_test_encoded = encode(autoencoder, data["X_test"])

        classifier = LogisticRegression(max_iter=5000, random_state=42)
        classifier.fit(X_train_encoded, data["y_train"])

        y_pred = classifier.predict(X_test_encoded)
        y_proba = classifier.predict_proba(X_test_encoded)[:, 1]

        acc = accuracy_score(data["y_test"], y_pred)
        f1 = f1_score(data["y_test"], y_pred)
        auc = roc_auc_score(data["y_test"], y_proba)
        print(f"bottleneck={bottleneck} -> Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}  (reconstruction MSE={recon_error:.4f})")

        if best_result is None or f1 > best_result["f1"]:
            best_result = {
                "bottleneck": bottleneck, "accuracy": acc, "f1": f1, "auc": auc,
                "precision": precision_score(data["y_test"], y_pred),
                "recall": recall_score(data["y_test"], y_pred),
            }

    print(f"\nΚαλύτερο bottleneck: {best_result['bottleneck']}")
    print(f"Accuracy:  {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-score:  {best_result['f1']:.4f}")
    print(f"ROC-AUC:   {best_result['auc']:.4f}")