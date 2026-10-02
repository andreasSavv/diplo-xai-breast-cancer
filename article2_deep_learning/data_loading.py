"""
data_loading.py

Φόρτωση του Breast Cancer Coimbra dataset (βιοδείκτες αίματος) απευθείας
από το UCI Machine Learning Repository.

116 δείγματα, 9 αριθμητικά χαρακτηριστικά (ηλικία, BMI, γλυκόζη, ινσουλίνη,
HOMA, λεπτίνη, αντιπονεκτίνη, ρεζιστίνη, MCP-1), 2 κλάσεις:
1 = υγιής (healthy control), 2 = ασθενής (breast cancer patient).
"""

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_coimbra_dataset(test_size=0.2, random_state=42, scale=True):
    dataset = fetch_ucirepo(id=451)  # Breast Cancer Coimbra

    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]  # 1 = healthy, 2 = patient

    # Μετατροπή σε 0/1 για ευκολία: 0 = healthy, 1 = patient
    y = y.map({1: 0, 2: 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": list(X.columns),
        "target_names": ["healthy", "patient"],
        "scaler": scaler,
    }


if __name__ == "__main__":
    d = load_coimbra_dataset()
    print(f"Train set: {d['X_train'].shape}")
    print(f"Test set:  {d['X_test'].shape}")
    print(f"Χαρακτηριστικά: {d['feature_names']}")
    print(f"Κατανομή train: {d['y_train'].value_counts().to_dict()}")