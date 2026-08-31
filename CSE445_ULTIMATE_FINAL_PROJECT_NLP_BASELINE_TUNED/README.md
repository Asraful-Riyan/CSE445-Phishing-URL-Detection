# CSE445 Ultimate Final Project - Phishing URL Detection

This repository combines the faculty final-report requirements with the agreed project design.

## ML experiments
Five base classifiers:
1. Logistic Regression
2. SVM
3. Decision Tree
4. Random Forest
5. XGBoost

After tuning, the five are ranked. The 2nd, 3rd and 4th best models become the base learners for both ensemble methods, while the best base classifier becomes the meta-learner.

Primary six-model table:
- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- XGBoost
- Stacking

Blending is additionally reported so both Stacking and Blending are demonstrated.

## Three augmentation techniques
- Random Oversampling
- SMOTE
- ADASYN

All augmentation is applied only to training data.

## Three NLP models from the lecture
- TF-IDF + Logistic Regression
- Word2Vec + Logistic Regression
- FastText + Logistic Regression

## Three contextual / LLM models from the lecture
- BERT
- GPT-family model (DistilGPT2)
- Grok API classifier

The Grok section runs only when `XAI_API_KEY` is supplied.

## Other work
- hyperparameter optimization
- regularization and early stopping
- train/test overfitting gap
- normalized confusion matrix
- model size, parameter count and training time
- LIME XAI
- ablation study
- KDD-inspired stability pipeline
- result CSV export

## Notebook
`notebooks/CSE445_ULTIMATE_FINAL_COLAB.ipynb`

Use a T4 GPU for BERT/GPT.
