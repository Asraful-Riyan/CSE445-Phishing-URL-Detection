import json
import joblib
import pandas as pd

from config import MODEL_DIR, RESULTS_DIR
from data_loader import load_dataset, split_dataset
from models import get_supervised_models, KMeansMappedClassifier
from evaluation import evaluate_predictions, get_phishing_probability


def train_and_evaluate():
    df, features = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df, features)

    print("=" * 70)
    print("CSE445 - PHISHING URL DETECTION")
    print("=" * 70)
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Input features: {len(features)}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print("Labels: 0 = Phishing, 1 = Legitimate")
    print()

    all_metrics = []
    trained_models = {}

    # Models 1-4: supervised tabular classifiers
    for name, model in get_supervised_models().items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = get_phishing_probability(model, X_test)

        metrics, report = evaluate_predictions(name, y_test, y_pred, score)
        all_metrics.append(metrics)
        trained_models[name] = model

        print(report)
        print("Metrics:", {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in metrics.items()})

        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    # Model 5: K-Means clustering baseline
    print("\nTraining K-Means Clustering ...")
    kmeans = KMeansMappedClassifier()
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_test)
    score = kmeans.phishing_score(X_test)
    metrics, report = evaluate_predictions("K-Means Clustering", y_test, y_pred, score)
    all_metrics.append(metrics)
    print(report)
    print("Cluster mapping:", kmeans.cluster_to_class)
    joblib.dump(kmeans, MODEL_DIR / "k_means_clustering.joblib")

    # Save model comparison for models 1-5.
    comparison = pd.DataFrame(all_metrics)
    comparison = comparison.sort_values(
        ["Recall_Phishing", "F1_Phishing", "ROC_AUC"],
        ascending=False
    ).reset_index(drop=True)

    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Models 1-5)")
    print("=" * 70)
    print(comparison.to_string(index=False))

    # Best real-time model chosen only from supervised models 1-4.
    supervised_names = set(trained_models.keys())
    supervised_results = comparison[comparison["Model"].isin(supervised_names)].copy()
    best_name = supervised_results.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    metadata = {
        "best_model": best_name,
        "features": features,
        "selection_rule": "Highest phishing recall, then F1, then ROC-AUC",
        "class_mapping": {"0": "Phishing", "1": "Legitimate"},
    }
    (MODEL_DIR / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nBest supervised real-time model: {best_name}")
    print("Saved: models/best_model.joblib")
    print("\nModel 6 (LLM) is evaluated separately with:")
    print("python src/llm_classifier.py --samples 200")

    return comparison


if __name__ == "__main__":
    train_and_evaluate()
