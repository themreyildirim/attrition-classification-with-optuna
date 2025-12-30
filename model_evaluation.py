"""
Final Model Evaluation
Trains models with optimized hyperparameters and evaluates performance.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Evaluate optimized models on HR Attrition data."""
    
    def __init__(self, data_path='data/WA_Fn-UseC_-HR-Employee-Attrition.csv',
                 params_path='output/best_hyperparameters.json'):
        """Initialize the evaluator."""
        self.data_path = data_path
        self.params_path = params_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.xgb_model = None
        self.lgb_model = None
        self.xgb_params = None
        self.lgb_params = None
        
    def load_best_params(self):
        """Load best hyperparameters from file."""
        if not os.path.exists(self.params_path):
            print(f"Warning: Parameters file not found at {self.params_path}")
            print("Using default parameters...")
            self.xgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            self.lgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'random_state': 42,
                'verbose': -1
            }
        else:
            with open(self.params_path, 'r') as f:
                params = json.load(f)
            
            self.xgb_params = params['xgboost']
            self.xgb_params.update({
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            })
            
            self.lgb_params = params['lightgbm']
            self.lgb_params.update({
                'random_state': 42,
                'verbose': -1
            })
            
            print("✓ Best hyperparameters loaded successfully!")
        
        return self
    
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
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self
    
    def train_models(self):
        """Train both XGBoost and LightGBM models."""
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(self.X_train, self.y_train)
        print("✓ XGBoost model trained successfully!")
        
        # Train LightGBM
        print("\nTraining LightGBM model...")
        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        self.lgb_model.fit(self.X_train, self.y_train)
        print("✓ LightGBM model trained successfully!")
        
        return self
    
    def evaluate_model(self, model, model_name):
        """Evaluate a single model."""
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} EVALUATION")
        print(f"{'='*80}")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Print metrics
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                    target_names=['No Attrition', 'Attrition']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def create_evaluation_visualizations(self, xgb_results, lgb_results, output_dir='output'):
        """Create visualizations for model evaluation."""
        print("\n" + "="*80)
        print("GENERATING EVALUATION VISUALIZATIONS")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Model Comparison - Metrics Bar Chart
        plt.figure(figsize=(12, 6))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        xgb_values = [xgb_results['accuracy'], xgb_results['precision'], 
                      xgb_results['recall'], xgb_results['f1'], xgb_results['roc_auc']]
        lgb_values = [lgb_results['accuracy'], lgb_results['precision'], 
                      lgb_results['recall'], lgb_results['f1'], lgb_results['roc_auc']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, xgb_values, width, label='XGBoost', color='steelblue')
        plt.bar(x + width/2, lgb_values, width, label='LightGBM', color='coral')
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/09_model_comparison.png")
        
        # 2. XGBoost Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(xgb_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Attrition', 'Attrition'],
                    yticklabels=['No Attrition', 'Attrition'])
        plt.title('XGBoost Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/10_xgb_confusion_matrix.png")
        
        # 3. LightGBM Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(lgb_results['confusion_matrix'], annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['No Attrition', 'Attrition'],
                    yticklabels=['No Attrition', 'Attrition'])
        plt.title('LightGBM Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/11_lgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/11_lgb_confusion_matrix.png")
        
        # 4. Feature Importance - XGBoost
        plt.figure(figsize=(10, 8))
        xgb_importances = pd.Series(
            self.xgb_model.feature_importances_,
            index=self.X_train.columns
        ).sort_values(ascending=False).head(15)
        
        xgb_importances.plot(kind='barh', color='steelblue')
        plt.title('Top 15 Feature Importances - XGBoost', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/12_xgb_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/12_xgb_feature_importance.png")
        
        # 5. Feature Importance - LightGBM
        plt.figure(figsize=(10, 8))
        lgb_importances = pd.Series(
            self.lgb_model.feature_importances_,
            index=self.X_train.columns
        ).sort_values(ascending=False).head(15)
        
        lgb_importances.plot(kind='barh', color='coral')
        plt.title('Top 15 Feature Importances - LightGBM', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/13_lgb_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/13_lgb_feature_importance.png")
    
    def save_results(self, xgb_results, lgb_results, output_dir='output'):
        """Save evaluation results to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'xgboost': {
                'accuracy': float(xgb_results['accuracy']),
                'precision': float(xgb_results['precision']),
                'recall': float(xgb_results['recall']),
                'f1_score': float(xgb_results['f1']),
                'roc_auc': float(xgb_results['roc_auc'])
            },
            'lightgbm': {
                'accuracy': float(lgb_results['accuracy']),
                'precision': float(lgb_results['precision']),
                'recall': float(lgb_results['recall']),
                'f1_score': float(lgb_results['f1']),
                'roc_auc': float(lgb_results['roc_auc'])
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_path = f'{output_dir}/evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Evaluation results saved to: {output_path}")

def main():
    """Main evaluation function."""
    print("="*80)
    print("HR ANALYTICS - FINAL MODEL EVALUATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load best parameters
    evaluator.load_best_params()
    
    # Load and preprocess data
    evaluator.load_and_preprocess_data()
    
    # Train models
    evaluator.train_models()
    
    # Evaluate XGBoost
    xgb_results = evaluator.evaluate_model(evaluator.xgb_model, 'XGBoost')
    
    # Evaluate LightGBM
    lgb_results = evaluator.evaluate_model(evaluator.lgb_model, 'LightGBM')
    
    # Create visualizations
    evaluator.create_evaluation_visualizations(xgb_results, lgb_results)
    
    # Save results
    evaluator.save_results(xgb_results, lgb_results)
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<15} {'XGBoost':<12} {'LightGBM':<12} {'Winner'}")
    print("-" * 55)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    
    for metric, name in zip(metrics, metric_names):
        xgb_val = xgb_results[metric]
        lgb_val = lgb_results[metric]
        winner = 'XGBoost' if xgb_val > lgb_val else 'LightGBM' if lgb_val > xgb_val else 'Tie'
        print(f"{name:<15} {xgb_val:<12.4f} {lgb_val:<12.4f} {winner}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check the 'output' directory for visualizations and results.")

if __name__ == "__main__":
    main()
