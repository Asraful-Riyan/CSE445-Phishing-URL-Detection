import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import RANDOM_STATE


def scaled_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def unscaled_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def get_supervised_models():
    return {
        "Logistic Regression": scaled_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "SVM": scaled_pipeline(
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": unscaled_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "XGBoost": unscaled_pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


class KMeansMappedClassifier:
    """
    K-Means baseline for binary classification.

    K-Means itself is unsupervised. After fitting two clusters on X_train,
    each cluster is mapped to the majority true class observed in X_train.
    """
    def __init__(self, random_state=RANDOM_STATE):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=2, n_init=20, random_state=random_state)),
        ])
        self.cluster_to_class = {}

    def fit(self, X, y):
        clusters = self.pipeline.fit_predict(X)
        y_array = np.asarray(y)

        for cluster_id in [0, 1]:
            values = y_array[clusters == cluster_id]
            if len(values) == 0:
                self.cluster_to_class[cluster_id] = 0
            else:
                counts = np.bincount(values.astype(int), minlength=2)
                self.cluster_to_class[cluster_id] = int(np.argmax(counts))
        return self

    def predict(self, X):
        clusters = self.pipeline.predict(X)
        return np.array([self.cluster_to_class[int(c)] for c in clusters])

    def phishing_score(self, X):
        """
        Produces a heuristic continuous score for ROC-AUC.
        Higher score means more phishing-like (class 0).
        """
        transformed = self.pipeline[:-1].transform(X)
        kmeans = self.pipeline.named_steps["model"]
        distances = kmeans.transform(transformed)

        phishing_clusters = [c for c, label in self.cluster_to_class.items() if label == 0]
        legit_clusters = [c for c, label in self.cluster_to_class.items() if label == 1]

        if not phishing_clusters or not legit_clusters:
            preds = self.predict(X)
            return (preds == 0).astype(float)

        pc = phishing_clusters[0]
        lc = legit_clusters[0]
        dp = distances[:, pc]
        dl = distances[:, lc]
        # Smaller distance to phishing cluster => larger phishing score.
        return dl / (dp + dl + 1e-12)
