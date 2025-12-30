#!/usr/bin/env python3
"""
Main pipeline runner for HR Analytics project.
Executes the complete workflow: data generation, EDA, tuning, and evaluation.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Running: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run the complete HR Analytics pipeline."""
    print("="*80)
    print("HR ANALYTICS COMPLETE PIPELINE")
    print("="*80)
    print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run the following steps:")
    print("  1. Generate synthetic HR dataset")
    print("  2. Exploratory Data Analysis (EDA)")
    print("  3. Hyperparameter Tuning with Optuna")
    print("  4. Final Model Evaluation")
    print("="*80)
    
    # Check if data exists
    if not os.path.exists('data/WA_Fn-UseC_-HR-Employee-Attrition.csv'):
        # Step 1: Generate data
        if not run_script('generate_data.py', 'STEP 1: Generating HR Dataset'):
            print("\n✗ Pipeline failed at data generation step!")
            return 1
    else:
        print("\n✓ Dataset already exists, skipping generation.")
    
    # Step 2: EDA
    if not run_script('exploratory_data_analysis.py', 'STEP 2: Exploratory Data Analysis'):
        print("\n✗ Pipeline failed at EDA step!")
        return 1
    
    # Step 3: Hyperparameter Tuning
    if not run_script('hyperparameter_tuning.py', 'STEP 3: Hyperparameter Tuning'):
        print("\n✗ Pipeline failed at hyperparameter tuning step!")
        return 1
    
    # Step 4: Model Evaluation
    if not run_script('model_evaluation.py', 'STEP 4: Model Evaluation'):
        print("\n✗ Pipeline failed at model evaluation step!")
        return 1
    
    # Success
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated outputs:")
    print("  - data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print("  - output/*.png (13 visualizations)")
    print("  - output/best_hyperparameters.json")
    print("  - output/evaluation_results.json")
    print("\nNext steps:")
    print("  - Review visualizations in the output/ directory")
    print("  - Check model performance in evaluation_results.json")
    print("  - Use trained models for predictions on new data")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
