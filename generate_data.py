"""
Generate a synthetic HR Analytics dataset for demonstration.
This mimics the structure of the IBM HR Analytics Employee Attrition dataset.
"""

import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=1470):
    """
    Generate a synthetic HR dataset similar to IBM HR Analytics dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of employee records to generate
    
    Returns:
    --------
    pd.DataFrame
        Synthetic HR dataset
    """
    
    np.random.seed(42)
    
    # Generate features
    data = {
        'Age': np.random.randint(18, 60, n_samples),
        'BusinessTravel': np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], n_samples, p=[0.15, 0.70, 0.15]),
        'DailyRate': np.random.randint(100, 1500, n_samples),
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples, p=[0.31, 0.65, 0.04]),
        'DistanceFromHome': np.random.randint(1, 30, n_samples),
        'Education': np.random.randint(1, 6, n_samples),
        'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'], n_samples),
        'EmployeeCount': np.ones(n_samples, dtype=int),
        'EmployeeNumber': range(1, n_samples + 1),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'HourlyRate': np.random.randint(30, 100, n_samples),
        'JobInvolvement': np.random.randint(1, 5, n_samples),
        'JobLevel': np.random.randint(1, 6, n_samples),
        'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                                      'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                      'Sales Representative', 'Research Director', 'Human Resources'], n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.32, 0.46, 0.22]),
        'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
        'MonthlyRate': np.random.randint(2000, 27000, n_samples),
        'NumCompaniesWorked': np.random.randint(0, 10, n_samples),
        'Over18': ['Y'] * n_samples,
        'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.28, 0.72]),
        'PercentSalaryHike': np.random.randint(11, 25, n_samples),
        'PerformanceRating': np.random.choice([3, 4], n_samples, p=[0.85, 0.15]),
        'RelationshipSatisfaction': np.random.randint(1, 5, n_samples),
        'StandardHours': np.ones(n_samples, dtype=int) * 80,
        'StockOptionLevel': np.random.randint(0, 4, n_samples),
        'TotalWorkingYears': np.random.randint(0, 40, n_samples),
        'TrainingTimesLastYear': np.random.randint(0, 7, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'YearsAtCompany': np.random.randint(0, 40, n_samples),
        'YearsInCurrentRole': np.random.randint(0, 18, n_samples),
        'YearsSinceLastPromotion': np.random.randint(0, 15, n_samples),
        'YearsWithCurrManager': np.random.randint(0, 17, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Adjust YearsAtCompany to be consistent with Age and TotalWorkingYears
    df['TotalWorkingYears'] = np.minimum(df['TotalWorkingYears'], df['Age'] - 18)
    df['YearsAtCompany'] = np.minimum(df['YearsAtCompany'], df['TotalWorkingYears'])
    df['YearsInCurrentRole'] = np.minimum(df['YearsInCurrentRole'], df['YearsAtCompany'])
    df['YearsSinceLastPromotion'] = np.minimum(df['YearsSinceLastPromotion'], df['YearsAtCompany'])
    df['YearsWithCurrManager'] = np.minimum(df['YearsWithCurrManager'], df['YearsAtCompany'])
    
    # Adjust MonthlyIncome based on JobLevel
    df['MonthlyIncome'] = 1000 + df['JobLevel'] * 3000 + np.random.randint(-500, 1500, n_samples)
    
    # Generate Attrition (target variable) with realistic patterns
    # Higher attrition for: lower satisfaction, overtime, frequent travel, lower income, etc.
    attrition_prob = 0.16  # Base attrition rate
    
    probs = np.ones(n_samples) * attrition_prob
    
    # Factors that increase attrition
    probs[df['OverTime'] == 'Yes'] *= 2.5
    probs[df['JobSatisfaction'] == 1] *= 2.0
    probs[df['EnvironmentSatisfaction'] == 1] *= 1.8
    probs[df['WorkLifeBalance'] == 1] *= 1.7
    probs[df['BusinessTravel'] == 'Travel_Frequently'] *= 1.5
    probs[df['YearsAtCompany'] < 2] *= 1.8
    probs[df['Age'] < 25] *= 1.4
    probs[df['DistanceFromHome'] > 20] *= 1.3
    
    # Factors that decrease attrition
    probs[df['JobSatisfaction'] == 4] *= 0.5
    probs[df['StockOptionLevel'] > 0] *= 0.7
    probs[df['YearsAtCompany'] > 10] *= 0.5
    
    # Cap probabilities
    probs = np.clip(probs, 0, 1)
    
    df['Attrition'] = np.random.binomial(1, probs, n_samples)
    df['Attrition'] = df['Attrition'].map({0: 'No', 1: 'Yes'})
    
    return df

def main():
    """Generate and save synthetic HR dataset."""
    
    print("Generating synthetic HR Analytics dataset...")
    df = generate_synthetic_data(n_samples=1470)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Dataset generated successfully!")
    print(f"  Saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Attrition rate: {(df['Attrition'] == 'Yes').sum() / len(df) * 100:.1f}%")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nColumn names:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
