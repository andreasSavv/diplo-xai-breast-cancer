"""
tabnet_model.py

TabNet - deep learning μοντέλο ειδικά σχεδιασμένο για tabular δεδομένα,
με ενσωματωμένο attention mechanism.

Σημείωση: με τόσο μικρό dataset (116 δείγματα συνολικά) χρησιμοποιούμε
μικρότερη χωρητικότητα (n_d, n_a) και περισσότερη υπομονή (patience)
ώστε να μην εγκαταλείπει η εκπαίδευση πολύ νωρίς.
"""

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    from data_loading import load_wdbc_dataset

    data = load_wdbc_dataset()

    X_train = data["X_train"].values.astype(np.float32)
    X_test = data["X_test"].values.astype(np.float32)
    y_train = data["y_train"].values.astype(np.int64)
    y_test = data["y_test"].values.astype(np.int64)

    model = TabNetClassifier(
        n_d=4,           # μικρότερη χωρητικότητα — κατάλληλο για λίγα δείγματα
        n_a=4,
        n_steps=2,       # λιγότερα βήματα attention
        gamma=1.3,
        lambda_sparse=1e-4,
        seed=42,
        verbose=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=["auc"],
        max_epochs=300,
        patience=60,     # πιο υπομονετικό early stopping
        batch_size=16,   # μικρό batch, κατάλληλο για 93 δείγματα εκπαίδευσης
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'=' * 60}\nTabNet - Τελικά αποτελέσματα\n{'=' * 60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print(f"\n{'=' * 60}\nΕνσωματωμένη σημαντικότητα χαρακτηριστικών (TabNet attention)\n{'=' * 60}")
    importances = model.feature_importances_
    feature_names = data["feature_names"]
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, importance in ranked:
        print(f"{name:15s}: {importance:.4f}")