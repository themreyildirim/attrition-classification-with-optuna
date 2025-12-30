"""
Data preparation script for HR Analytics dataset.
Downloads the IBM HR Analytics Employee Attrition & Performance dataset.
"""

import pandas as pd
import os

def prepare_data():
    """
    Prepare the HR Analytics dataset.
    For this implementation, we'll create a sample dataset with the typical structure
    of HR attrition data that would be used in practice.
    """
    
    # Define the data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Setting up HR Analytics dataset structure...")
    print(f"Data will be stored in: {os.path.abspath(data_dir)}")
    print("\nNote: In a real project, you would download the IBM HR Analytics dataset from:")
    print("https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
    print("\nFor this implementation, you should place 'WA_Fn-UseC_-HR-Employee-Attrition.csv'")
    print(f"in the {data_dir}/ directory.")
    
    # Check if data file exists
    data_file = os.path.join(data_dir, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if os.path.exists(data_file):
        print(f"\n✓ Dataset found at: {data_file}")
        df = pd.read_csv(data_file)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        return df
    else:
        print(f"\n✗ Dataset not found at: {data_file}")
        print("  Please download the dataset and place it in the data/ directory.")
        return None

if __name__ == "__main__":
    prepare_data()
