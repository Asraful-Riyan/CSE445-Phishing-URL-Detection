import argparse
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import TARGET_COLUMN, RESULTS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader import load_dataset
from evaluation import evaluate_predictions


DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"


def row_to_text(row, feature_columns):
    parts = []
    for col in feature_columns:
        value = row[col]
        if pd.isna(value):
            value = "missing"
        parts.append(f"{col}={value}")
    return ", ".join(parts)


def build_prompt(feature_text):
    return (
        "You are a cybersecurity classifier. "
        "Classify the URL represented by these tabular lexical and structural features. "
        "Reply with exactly one word: phishing or legitimate.\n\n"
        f"Features: {feature_text}\n"
        "Answer:"
    )


def parse_label(text):
    clean = text.strip().lower()
    # Important: dataset mapping is 0=phishing, 1=legitimate.
    if "phishing" in clean:
        return 0
    if "legitimate" in clean or "benign" in clean or "safe" in clean:
        return 1
    return None


def run_llm_evaluation(samples=200, model_name=DEFAULT_LLM):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError as exc:
        raise ImportError(
            "LLM packages are missing. Run: pip install transformers torch accelerate"
        ) from exc

    df, features = load_dataset()

    # Use the same stratified 20% test concept, then sample from test for practical LLM runtime.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    n = min(samples, len(test_df))
    test_sample = test_df.sample(
        n=n,
        random_state=RANDOM_STATE,
        weights=None
    ).copy()

    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not torch.cuda.is_available():
        model = model.to("cpu")

    predictions = []
    raw_outputs = []

    for i, (_, row) in enumerate(test_sample.iterrows(), 1):
        feature_text = row_to_text(row, features)
        prompt = build_prompt(feature_text)

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            rendered = prompt

        inputs = tokenizer(rendered, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        pred = parse_label(answer)

        # Conservative fallback if the model does not follow the one-word format.
        if pred is None:
            pred = 0

        predictions.append(pred)
        raw_outputs.append(answer)

        if i % 20 == 0 or i == n:
            print(f"Processed {i}/{n}")

    y_true = test_sample[TARGET_COLUMN].to_numpy()
    y_pred = np.asarray(predictions)

    metrics, report = evaluate_predictions(
        "LLM",
        y_true,
        y_pred,
        phishing_score=None,
        save_plots=True
    )

    out = test_sample.copy()
    out["LLM_raw_output"] = raw_outputs
    out["LLM_prediction"] = y_pred
    out["LLM_prediction_name"] = np.where(y_pred == 0, "Phishing", "Legitimate")
    out.to_csv(RESULTS_DIR / "llm_predictions.csv", index=False)

    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "llm_results.csv", index=False)

    print("\nLLM Classification Report")
    print(report)
    print("\nLLM Metrics")
    print(metrics)
    print("\nSaved results/llm_results.csv and results/llm_predictions.csv")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_LLM)
    args = parser.parse_args()
    run_llm_evaluation(samples=args.samples, model_name=args.model)
