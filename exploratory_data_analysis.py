"""
Exploratory Data Analysis (EDA) for HR Analytics Dataset
Analyzes employee attrition patterns and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(data_path='data/WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """Load the HR Analytics dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    return df

def basic_statistics(df):
    """Display basic statistics about the dataset."""
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Data Types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print(missing[missing > 0])
    
    print(f"\nAttrition Distribution:")
    attrition_counts = df['Attrition'].value_counts()
    print(attrition_counts)
    print(f"\nAttrition Rate: {attrition_counts['Yes'] / len(df) * 100:.2f}%")
    
    return df

def analyze_attrition_by_features(df):
    """Analyze attrition rates across different features."""
    print("\n" + "="*80)
    print("ATTRITION ANALYSIS BY FEATURES")
    print("="*80)
    
    # Categorical features to analyze
    categorical_features = [
        'Department', 'Gender', 'JobRole', 'MaritalStatus', 
        'BusinessTravel', 'OverTime', 'EducationField'
    ]
    
    for feature in categorical_features:
        if feature in df.columns:
            print(f"\n{feature}:")
            attrition_by_feature = pd.crosstab(df[feature], df['Attrition'], normalize='index') * 100
            print(attrition_by_feature.round(2))

def analyze_numerical_features(df):
    """Analyze numerical features and their relationship with attrition."""
    print("\n" + "="*80)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nNumerical Features Summary Statistics:")
    print(df[numerical_cols].describe())
    
    # Compare numerical features between attrition groups
    print(f"\nNumerical Features by Attrition Status:")
    for col in ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 
                'TotalWorkingYears', 'JobSatisfaction', 'EnvironmentSatisfaction']:
        if col in df.columns:
            print(f"\n{col}:")
            print(df.groupby('Attrition')[col].describe()[['mean', 'std', 'min', 'max']])

def create_visualizations(df, output_dir='output'):
    """Create and save visualizations."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Attrition Distribution
    plt.figure(figsize=(8, 6))
    df['Attrition'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Employee Attrition Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Attrition')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_attrition_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/01_attrition_distribution.png")
    
    # 2. Attrition by Department
    plt.figure(figsize=(10, 6))
    dept_attrition = pd.crosstab(df['Department'], df['Attrition'])
    dept_attrition.plot(kind='bar', stacked=False, color=['green', 'red'])
    plt.title('Attrition by Department', fontsize=14, fontweight='bold')
    plt.xlabel('Department')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Attrition')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_attrition_by_department.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/02_attrition_by_department.png")
    
    # 3. Attrition by OverTime
    plt.figure(figsize=(8, 6))
    overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
    overtime_attrition.plot(kind='bar', color=['green', 'red'])
    plt.title('Attrition Rate by OverTime', fontsize=14, fontweight='bold')
    plt.xlabel('OverTime')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Attrition')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_attrition_by_overtime.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/03_attrition_by_overtime.png")
    
    # 4. Age Distribution by Attrition
    plt.figure(figsize=(10, 6))
    for attrition_status in ['No', 'Yes']:
        data = df[df['Attrition'] == attrition_status]['Age']
        plt.hist(data, alpha=0.6, label=attrition_status, bins=20)
    plt.title('Age Distribution by Attrition Status', fontsize=14, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(title='Attrition')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/04_age_distribution.png")
    
    # 5. Monthly Income by Attrition
    plt.figure(figsize=(10, 6))
    df.boxplot(column='MonthlyIncome', by='Attrition', patch_artist=True)
    plt.title('Monthly Income by Attrition Status', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Attrition')
    plt.ylabel('Monthly Income')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_income_by_attrition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/05_income_by_attrition.png")
    
    # 6. Years at Company by Attrition
    plt.figure(figsize=(10, 6))
    df.boxplot(column='YearsAtCompany', by='Attrition', patch_artist=True)
    plt.title('Years at Company by Attrition Status', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Attrition')
    plt.ylabel('Years at Company')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_years_at_company.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/06_years_at_company.png")
    
    # 7. Correlation Heatmap (numerical features)
    plt.figure(figsize=(14, 10))
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Select key features for correlation
    key_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
                    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                    'JobLevel', 'StockOptionLevel', 'JobSatisfaction', 
                    'EnvironmentSatisfaction', 'WorkLifeBalance', 'JobInvolvement']
    key_features = [f for f in key_features if f in df.columns]
    
    corr_matrix = df[key_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Key Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/07_correlation_heatmap.png")
    
    # 8. Job Satisfaction vs Attrition
    plt.figure(figsize=(10, 6))
    satisfaction_attrition = pd.crosstab(df['JobSatisfaction'], df['Attrition'], normalize='index') * 100
    satisfaction_attrition.plot(kind='bar', color=['green', 'red'])
    plt.title('Attrition Rate by Job Satisfaction Level', fontsize=14, fontweight='bold')
    plt.xlabel('Job Satisfaction (1=Low, 4=High)')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Attrition')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_job_satisfaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/08_job_satisfaction.png")

def main():
    """Main EDA function."""
    print("="*80)
    print("HR ANALYTICS - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    
    # Basic statistics
    df = basic_statistics(df)
    
    # Analyze attrition by features
    analyze_attrition_by_features(df)
    
    # Analyze numerical features
    analyze_numerical_features(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check the 'output' directory for generated visualizations.")

if __name__ == "__main__":
    main()
