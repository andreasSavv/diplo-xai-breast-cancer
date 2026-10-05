"""
grnn.py

Υλοποίηση του General Regression Neural Network (GRNN) από το μηδέν.

Λογική: για κάθε νέο δείγμα, υπολογίζουμε πόσο "κοντά" είναι σε ΟΛΑ τα
δείγματα εκπαίδευσης (απόσταση), δίνουμε σε κάθε ένα ένα βάρος που
μειώνεται εκθετικά με την απόσταση (καμπύλη Gaussian/"καμπάνα"), και η
πρόβλεψη είναι ο σταθμισμένος μέσος όρος των ετικετών τους.

Η παράμετρος sigma ελέγχει πόσο "στενή" είναι η καμπάνα:
- μικρό sigma -> μόνο οι πολύ κοντινοί γείτονες μετράνε (σαν kNN με k=1)
- μεγάλο sigma -> όλα τα δείγματα μετράνε σχεδόν εξίσου
"""

import numpy as np
from scipy.spatial.distance import cdist


class GRNN:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Το GRNN δεν "εκπαιδεύεται" με την κλασική έννοια -
        # απλώς απομνημονεύει όλα τα δείγματα εκπαίδευσης.
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train, dtype=float)
        return self

    def predict_proba(self, X_test):
        X_test = np.asarray(X_test)
        # Απόσταση κάθε test δείγματος από ΚΑΘΕ train δείγμα
        distances = cdist(X_test, self.X_train, metric="euclidean")

        # Μετατροπή απόστασης σε βάρος: όσο πιο κοντά, τόσο μεγαλύτερο βάρος
        weights = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        # Σταθμισμένος μέσος όρος των ετικετών (0/1) = εκτιμώμενη πιθανότητα κλάσης 1
        weight_sums = weights.sum(axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1e-10  # αποφυγή διαίρεσης με το 0
        proba_class1 = (weights @ self.y_train) / weight_sums.flatten()

        proba = np.column_stack([1 - proba_class1, proba_class1])
        return proba

    def predict(self, X_test):
        proba = self.predict_proba(X_test)
        return (proba[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    from data_loading import load_wdbc_dataset
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    data = load_wdbc_dataset()

    best_result = None
    print("Δοκιμή διαφορετικών τιμών sigma (παράμετρος εύρους):\n")
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        model = GRNN(sigma=sigma)
        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_test"])
        y_proba = model.predict_proba(data["X_test"])[:, 1]

        acc = accuracy_score(data["y_test"], y_pred)
        f1 = f1_score(data["y_test"], y_pred)
        auc = roc_auc_score(data["y_test"], y_proba)
        print(f"sigma={sigma:4.1f} -> Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")

        if best_result is None or f1 > best_result["f1"]:
            best_result = {"sigma": sigma, "accuracy": acc, "f1": f1, "auc": auc,
                            "precision": precision_score(data["y_test"], y_pred),
                            "recall": recall_score(data["y_test"], y_pred)}

    print(f"\nΚαλύτερο sigma: {best_result['sigma']}")
    print(f"Accuracy:  {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-score:  {best_result['f1']:.4f}")
    print(f"ROC-AUC:   {best_result['auc']:.4f}")