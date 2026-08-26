import argparse
import json
import joblib
import pandas as pd
from config import MODEL_DIR
from feature_extractor import extract_url_features


def predict(url):
    model_path = MODEL_DIR / "best_model.joblib"
    meta_path = MODEL_DIR / "best_model_metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python run_project.py` first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    features = metadata["features"]

    row = extract_url_features(url)
    X = pd.DataFrame([[row.get(f, 0) for f in features]], columns=features)

    pred = int(model.predict(X)[0])
    label = "PHISHING" if pred == 0 else "LEGITIMATE"

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        confidence = float(probs[classes.index(pred)])

    print("\nURL:", url)
    print("Prediction:", label)
    if confidence is not None:
        print(f"Confidence: {confidence * 100:.2f}%")
    print("Model:", metadata["best_model"])

    return pred, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to classify")
    args = parser.parse_args()
    predict(args.url)
