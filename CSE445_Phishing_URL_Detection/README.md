# CSE445 - Real-Time Phishing URL Detection Using Machine Learning

This project implements the CSE445 proposal **Real-Time Phishing URL Detection using Machine Learning** using the **LegitPhish Dataset (Version 2, 2025)**.

## Dataset

Primary dataset:
- Name: LegitPhish Dataset
- File: `url_features_extracted1.csv`
- Rows: 101,219 URLs
- Columns: 18 including the target
- Target: `ClassLabel`
- `0 = Phishing`
- `1 = Legitimate`
- Source: Mendeley Data, DOI `10.17632/hx4m73v2sf.2`

Download the CSV and place it here:

`data/url_features_extracted1.csv`

## Six Models

The project compares exactly six approaches:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost
5. K-Means Clustering
6. Large Language Model (LLM) zero-shot tabular classifier

### Important note about K-Means

K-Means is an unsupervised clustering algorithm, not a normal supervised classifier.
The code fits two clusters and maps each cluster to the majority class in the training set.
This allows a fair baseline comparison with the classification models.

### Important note about the LLM

The LLM converts each row of tabular URL features into a short text description and asks an instruction-tuned language model to classify it as `phishing` or `legitimate`.

Default model:
`Qwen/Qwen2.5-0.5B-Instruct`

Because LLM inference is much slower than traditional tabular ML, it is evaluated on a configurable subset of the test data by default.

## Evaluation

The project reports:
- Accuracy
- Precision for phishing
- Recall for phishing
- F1-score for phishing
- ROC-AUC
- Confusion Matrix
- Classification Report

The proposal prioritizes **high phishing recall** because missed phishing URLs are especially dangerous.

## Project Structure

```text
CSE445_Phishing_URL_Detection/
│
├── data/
│   └── PUT_DATASET_HERE.txt
├── models/
├── results/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_extractor.py
│   ├── llm_classifier.py
│   ├── models.py
│   ├── predict_url.py
│   └── train_all.py
├── notebooks/
│   └── CSE445_Phishing_URL_Detection.ipynb
├── .gitignore
├── requirements.txt
├── run_project.py
└── README.md
```

## Run Locally

### 1. Install packages

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Place:

```text
url_features_extracted1.csv
```

inside:

```text
data/
```

### 3. Train and evaluate the five numeric/tabular models

```bash
python run_project.py
```

This trains:
- Logistic Regression
- SVM
- Random Forest
- XGBoost
- K-Means

and saves metrics, plots, and the best supervised model.

### 4. Run LLM evaluation

```bash
python src/llm_classifier.py
```

You can change the number of test rows:

```bash
python src/llm_classifier.py --samples 200
```

### 5. Predict a new URL in real time

After training:

```bash
python src/predict_url.py "https://example.com/login"
```

## Google Colab

Open:

`notebooks/CSE445_Phishing_URL_Detection.ipynb`

Upload the dataset when requested and run the cells from top to bottom.

## Output Files

The `results/` folder will contain:
- `model_comparison.csv`
- confusion matrix PNG files
- ROC curve PNG files
- classification report TXT files
- `llm_results.csv` after LLM evaluation

The `models/` folder will contain:
- trained model files
- `best_model.joblib`
- best model metadata

## Proposed Pipeline

```text
URL / Dataset
      |
      v
Data Cleaning
      |
      v
Feature Selection / Scaling
      |
      +-----------------------------+
      |                             |
      v                             v
Supervised ML                   K-Means
LR / SVM / RF / XGB             Clustering
      |                             |
      +-------------+---------------+
                    |
                    v
          Performance Evaluation
                    |
                    v
     Accuracy / Precision / Recall
          F1 / ROC-AUC / CM

Tabular Row
    |
    v
Feature-to-Text Conversion
    |
    v
Instruction LLM
    |
    v
Phishing / Legitimate
```

## Academic Note

The LLM and K-Means are included as comparison approaches. For a tabular binary classification problem, Random Forest, XGBoost, SVM, and Logistic Regression are more conventional choices. K-Means provides an unsupervised baseline, while the LLM demonstrates how structured features can be represented as text for language-model inference.
