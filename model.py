"""
Credit Card Fraud Detection - Ensemble Learning Pipeline
Combines EllipticEnvelope anomaly detection with LightGBM gradient boosting,
optimized via Optuna Bayesian hyperparameter search for extreme class imbalance (588:1).

Author: Seohyun Park
Date: October 2025
License: MIT
GitHub: https://github.com/spbraden2007-ux/Credit_Card_Fraud_Detection_Model

Technical Details:
- Dataset: 284,807 transactions (0.17% fraud rate) from Dacon competition
  Competition URL: https://dacon.io/competitions/official/235930/data 
- Method: Pseudo-labeling via anomaly scores → supervised learning → ensemble voting
- Performance: 0.93 F1-macro (Dacon Public Leaderboard), 0.87 CV F1-macro (5-fold)
- Optimization: 500 Optuna trials targeting macro F1-score
"""

import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import EllipticEnvelope
from tqdm.auto import tqdm
import lightgbm
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import torch
import warnings
warnings.filterwarnings(action='ignore')

# ============================================================================
# STEP 1: Data Loading
# ============================================================================
# Load preprocessed data with PCA-transformed features (V1-V28)
train = pd.read_csv('Credit_Card_Fraud_Detection_Model/open/train.csv') 
val = pd.read_csv('Credit_Card_Fraud_Detection_Model/open/val.csv')
test = pd.read_csv('Credit_Card_Fraud_Detection_Model/open/test.csv')

# Remove ID column (not used for modeling)
trainset = train.drop(['ID'], axis=1) 
testset = test.drop(['ID'], axis=1) 

# Calculate fraud ratio from validation set (0.17% = 1:588 imbalance)
fraud_ratio = val['Class'].values.sum() / len(val)
print(f"Fraud ratio: {fraud_ratio:.4f}")

# ============================================================================
# STEP 2: Unsupervised Anomaly Detection with EllipticEnvelope
# ============================================================================
# Fit Gaussian distribution to identify outliers (potential fraud cases)
# support_fraction: proportion of inliers used to compute robust estimates
# contamination: expected proportion of outliers (fraud) in the dataset
model = EllipticEnvelope(
    support_fraction=0.994, 
    contamination=fraud_ratio, 
    random_state=42
) 
model.fit(trainset)

# ============================================================================
# STEP 3: Pseudo-Label Generation Function
# ============================================================================
def get_pred_label(model, x, k):
    """
    Generate pseudo-labels by selecting top-k most anomalous samples.
    
    Args:
        model: Fitted EllipticEnvelope model
        x: Feature matrix (DataFrame or array)
        k: Number of samples to label as fraud (top-k most anomalous)
    
    Returns:
        pred: Binary labels (1=fraud, 0=normal)
        prob: Anomaly scores (lower = more anomalous)
    
    Strategy: Use anomaly scores to create pseudo-labels for supervised learning.
    Lower scores indicate higher anomaly likelihood (potential fraud).
    """
    prob = model.score_samples(x)
    prob = torch.tensor(prob, dtype=torch.float)
    # Select top-k samples with lowest anomaly scores (most suspicious)
    topk_indices = torch.topk(prob, k=k, largest=False).indices

    pred = torch.zeros(len(x), dtype=torch.long)
    pred[topk_indices] = 1  # Label top-k as fraud
    return pred, prob

# ============================================================================
# STEP 4: Generate Pseudo-Labels for Test and Train Sets
# ============================================================================
# Test set: k=314 based on empirical validation (observed 305-325 fraud cases)
# Using middle value (313-314) for stability
test_pred, _ = get_pred_label(model, testset, k=314)
envelope_pred = np.array(test_pred)

# Train set: k=118 maintains similar fraud ratio as validation set
# This creates pseudo-labels for supervised LightGBM training
train_pred, _ = get_pred_label(model, trainset, k=118)
X = np.array(train_pred)  # Pseudo-labels for training

# ============================================================================
# STEP 5: Optuna Hyperparameter Optimization
# ============================================================================
def lgb_optimization(trial):
    """
    Optuna objective function for LightGBM hyperparameter tuning.
    
    Optimization goal: Maximize macro F1-score (balances precision/recall for both classes)
    Strategy: 5-fold stratified cross-validation to ensure robust parameter selection
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
    
    Returns:
        Mean macro F1-score across 5 folds
    """
    score = []
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    
    for train_fold, test_fold in tqdm(skf.split(trainset, X), desc='k_fold'):
        X_train, X_test = trainset.iloc[train_fold], trainset.iloc[test_fold]
        y_train, y_test = X[train_fold], X[test_fold]
        
        # Hyperparameter search space
        params = {            
            "boosting_type": trial.suggest_categorical('boosting_type', ['dart', 'gbdt']),
            "learning_rate": trial.suggest_uniform('learning_rate', 0.2, 0.99),
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=10),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1),  # L1 regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1),  # L2 regularization
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),  # Row sampling ratio
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 30),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),  # Feature sampling
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "max_bin": trial.suggest_int("max_bin", 50, 100),  # Histogram bins
            "verbosity": -1,
            "random_state": trial.suggest_int("random_state", 1, 10000)
        }
        
        # Train LightGBM on current fold
        model_lgb = LGBMClassifier(**params)
        model_lgb.fit(X_train, y_train)
        lgb_cv_pred = model_lgb.predict(X_test)
        
        # Evaluate using macro F1 (unweighted mean of F1 for each class)
        score_cv = f1_score(y_test, lgb_cv_pred, average='macro')
        score.append(score_cv)
    
    print(f"Trial scores: {score}")
    return np.mean(score)

# ============================================================================
# STEP 6: Run Bayesian Optimization (500 trials)
# ============================================================================
# TPE (Tree-structured Parzen Estimator) sampler for efficient search
sampler = TPESampler()
optim = optuna.create_study(
    study_name="lgb_parameter_opt",
    direction="maximize",  # Maximize macro F1-score
    sampler=sampler,
)

print("Starting Optuna hyperparameter optimization (500 trials)...")
optim.optimize(lgb_optimization, n_trials=500)
print(f"Best macro-F1: {optim.best_value:.4f}")
print(f"Best parameters: {optim.best_params}")

# ============================================================================
# STEP 7: Train Final Model with Optimized Hyperparameters
# ============================================================================
# Best parameters found through multiple optimization runs
# These values provide stable performance across different random seeds
params = {
    'boosting_type': 'dart',  # Dropout Additive Regression Trees
    'learning_rate': 0.3066049775331286,
    'n_estimators': 270,
    'max_depth': 7,
    'num_leaves': 66,
    'reg_alpha': 0.053095543407827614,  # L1 regularization
    'reg_lambda': 0.8491712589623094,   # L2 regularization (stronger)
    'subsample': 0.5662898983569683,    # 56% row sampling
    'subsample_freq': 1,
    'colsample_bytree': 0.9078745461941586,  # 91% feature sampling
    'min_child_samples': 31,
    'max_bin': 52
}

# Train final LightGBM model on full training set
model2 = LGBMClassifier(**params, random_state=2893)
model2.fit(trainset, X)
lgb_pred = model2.predict(testset)

# ============================================================================
# STEP 8: Ensemble Voting (Logical OR)
# ============================================================================
# Combine predictions: flag as fraud if EITHER model predicts fraud
# Strategy: Maximize recall (catch more fraud) at cost of precision
# Trade-off: False positives cheaper than false negatives in fraud detection
sub = pd.read_csv('Credit_Card_Fraud_Detection_Model/open/sample_submission.csv')
sub['Class'] = envelope_pred | lgb_pred  # Logical OR ensemble
sub.to_csv('Credit_Card_Fraud_Detection_Model/save/result.csv', index=False)

print(f"✓ Predictions saved to Credit_Card_Fraud_Detection_Model/save/result.csv")
print(f"✓ Total fraud cases detected: {sub['Class'].sum()}")