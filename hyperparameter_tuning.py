"""
Hyperparameter Tuning using Optuna
Optimizes XGBoost and LightGBM models for employee attrition prediction.
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class HRAttritionOptimizer:
    """Hyperparameter optimization for HR Attrition models."""
    
    def __init__(self, data_path='data/WA_Fn-UseC_-HR-Employee-Attrition.csv'):
        """Initialize the optimizer."""
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.best_xgb_params = None
        self.best_lgb_params = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the HR dataset."""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Drop columns that are not useful for prediction
        columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        
        # Separate features and target
        X = df.drop('Attrition', axis=1)
        y = df['Attrition'].map({'No': 0, 'Yes': 1})
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set attrition rate: {self.y_train.mean():.2%}")
        print(f"Test set attrition rate: {self.y_test.mean():.2%}")
        
        return self
    
    def objective_xgboost(self, trial):
        """Optuna objective function for XGBoost."""
        
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                                 cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()
    
    def objective_lightgbm(self, trial):
        """Optuna objective function for LightGBM."""
        
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        # Create model
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                                 cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()
    
    def optimize_xgboost(self, n_trials=50):
        """Optimize XGBoost hyperparameters."""
        print("\n" + "="*80)
        print("OPTIMIZING XGBOOST")
        print("="*80)
        print(f"Running {n_trials} trials...")
        
        study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
        study.optimize(self.objective_xgboost, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest trial:")
        print(f"  Value (ROC-AUC): {study.best_trial.value:.4f}")
        print(f"  Parameters: {study.best_trial.params}")
        
        self.best_xgb_params = study.best_trial.params
        return study
    
    def optimize_lightgbm(self, n_trials=50):
        """Optimize LightGBM hyperparameters."""
        print("\n" + "="*80)
        print("OPTIMIZING LIGHTGBM")
        print("="*80)
        print(f"Running {n_trials} trials...")
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
        study.optimize(self.objective_lightgbm, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest trial:")
        print(f"  Value (ROC-AUC): {study.best_trial.value:.4f}")
        print(f"  Parameters: {study.best_trial.params}")
        
        self.best_lgb_params = study.best_trial.params
        return study
    
    def save_best_params(self, output_dir='output'):
        """Save best hyperparameters to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        params = {
            'xgboost': self.best_xgb_params,
            'lightgbm': self.best_lgb_params,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_path = f'{output_dir}/best_hyperparameters.json'
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"\n✓ Best hyperparameters saved to: {output_path}")

def main():
    """Main hyperparameter tuning function."""
    print("="*80)
    print("HR ANALYTICS - HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize optimizer
    optimizer = HRAttritionOptimizer()
    
    # Load and preprocess data
    optimizer.load_and_preprocess_data()
    
    # Optimize XGBoost
    xgb_study = optimizer.optimize_xgboost(n_trials=50)
    
    # Optimize LightGBM
    lgb_study = optimizer.optimize_lightgbm(n_trials=50)
    
    # Save best parameters
    optimizer.save_best_params()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest XGBoost ROC-AUC: {xgb_study.best_trial.value:.4f}")
    print(f"Best LightGBM ROC-AUC: {lgb_study.best_trial.value:.4f}")

if __name__ == "__main__":
    main()
