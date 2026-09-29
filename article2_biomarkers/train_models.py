"""
train_models.py

Εκπαίδευση και αξιολόγηση των 4 μοντέλων του Άρθρου 2:
- Logistic Regression
- kNN (k-Nearest Neighbors)
- SVM (RBF kernel)
- RUSBoost (boosting με random under-sampling για ανισόρροπες κλάσεις)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.ensemble import RUSBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import pandas as pd


def get_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=random_state),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "RUSBoost": RUSBoostClassifier(n_estimators=100, random_state=random_state),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def train_and_evaluate_all(data, random_state=42, verbose=True):
    models = get_models(random_state=random_state)
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(data["X_train"], data["y_train"])
        metrics = evaluate_model(model, data["X_test"], data["y_test"])
        results[name] = metrics
        trained_models[name] = model

        if verbose:
            print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-score:  {metrics['f1']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    summary_table = pd.DataFrame({
        name: {
            "Accuracy": r["accuracy"], "Precision": r["precision"],
            "Recall": r["recall"], "F1-score": r["f1"], "ROC-AUC": r["roc_auc"],
        }
        for name, r in results.items()
    }).T.sort_values("F1-score", ascending=False)

    return trained_models, results, summary_table


if __name__ == "__main__":
    from data_loading import load_coimbra_dataset

    data = load_coimbra_dataset()
    trained_models, results, summary_table = train_and_evaluate_all(data)

    print(f"\n{'=' * 60}\nΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ\n{'=' * 60}")
    print(summary_table.round(4).to_string())