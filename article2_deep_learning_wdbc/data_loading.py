from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_wdbc_dataset(test_size=0.2, random_state=42, scale=True):
    """
    Wisconsin Diagnostic Breast Cancer dataset.
    569 δείγματα, 30 χαρακτηριστικά (εξαγόμενα από ψηφιακές εικόνες
    λεπτής βελονικής παρακέντησης - FNA). 0 = κακοήθης, 1 = καλοήθης.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # ήδη 0/1

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
        "target_names": ["malignant", "benign"],
        "scaler": scaler,
    }