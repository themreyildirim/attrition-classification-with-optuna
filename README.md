# Employee Attrition Classification with Optuna

A machine learning project that predicts employee attrition using an ensemble of XGBoost and LightGBM models, with hyperparameter optimization powered by Optuna. This project was developed for the Kaggle Playground Series Season 3, Episode 3 competition.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features & Data Preprocessing](#features--data-preprocessing)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Project Overview

Employee attrition prediction is a critical task for HR analytics and workforce planning. This project implements a comprehensive machine learning pipeline to predict whether an employee will leave the company (attrition) based on various employee attributes and workplace factors.

The solution employs advanced techniques including:
- **Feature engineering**: Creating new meaningful features from existing data
- **Outlier capping**: Handling extreme values in numerical features
- **Log transformation**: Normalizing skewed distributions
- **Encoding**: Both ordinal and one-hot encoding for categorical variables
- **Hyperparameter optimization**: Using Optuna for efficient hyperparameter tuning
- **Ensemble learning**: Combining XGBoost and LightGBM predictions

## Dataset

**Source**: [Kaggle Playground Series - Season 3, Episode 3](https://www.kaggle.com/competitions/playground-series-s3e3)

The dataset contains employee information with 34 features including:
- **Demographics**: Age, Gender, MaritalStatus, DistanceFromHome
- **Job Information**: JobRole, JobLevel, Department, BusinessTravel
- **Compensation**: MonthlyIncome, HourlyRate, DailyRate, MonthlyRate, StockOptionLevel
- **Work History**: TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager
- **Satisfaction Metrics**: JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance
- **Performance**: PerformanceRating, PercentSalaryHike
- **Other**: Education, EducationField, TrainingTimesLastYear

**Class Distribution**: The dataset is imbalanced with a negative-to-positive ratio of approximately 7.38:1 (more employees staying than leaving).

**Dataset Size**:
- Training set: ~3,000 samples
- Test set: 1,119 samples

## Features & Data Preprocessing

### 1. **Data Cleaning**
- Removed constant features: `Over18`, `EmployeeCount`, `StandardHours`
- Handled missing values and data quality issues

### 2. **Feature Engineering**
Created new features to capture relationships:
- `AverageTenurePerCompany`: Calculates average years per company (TotalWorkingYears / NumCompaniesWorked)
- `Role_Stagnation_Ratio`: Measures career stagnation (YearsInCurrentRole / YearsAtCompany)
- `IncomePerAge`: Ratio of monthly income to age
- `IncomePerJobLevel`: Income normalized by job level

### 3. **Outlier Treatment**
Applied outlier capping using IQR method (3x multiplier) for numerical features to reduce the impact of extreme values.

### 4. **Transformation**
- **Log transformation** applied to skewed features:
  - YearsSinceLastPromotion
  - YearsAtCompany
  - MonthlyIncome
  - TotalWorkingYears
  - YearsInCurrentRole
  - YearsWithCurrManager

### 5. **Encoding**
- **Ordinal Encoding**: BusinessTravel, OverTime
- **One-Hot Encoding**: Department, EducationField, Gender, JobRole, MaritalStatus

### 6. **Feature Selection**
- Removed high VIF (Variance Inflation Factor) features to reduce multicollinearity
- Dropped features with low importance after initial analysis

## Model Architecture

### Hyperparameter Optimization with Optuna

**Optuna** is used to efficiently search the hyperparameter space for both models:

#### XGBoost Configuration
- **Search Space**:
  - n_estimators: 100-1000
  - max_depth: 3-10
  - learning_rate: 0.001-0.1 (log scale)
  - subsample: 0.5-1.0
  - colsample_bytree: 0.5-1.0
  - gamma: 0-5
  - min_child_weight: 1-10
  - scale_pos_weight: 1-11 (to handle class imbalance)

- **Trials**: 20 optimization trials
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Evaluation Metric**: ROC-AUC

#### LightGBM Configuration
- **Search Space**:
  - n_estimators: 100-300
  - max_depth: 3-7
  - learning_rate: 0.05-0.2
  - num_leaves: 20-30
  - min_child_samples: 20-50
  - scale_pos_weight: 1-10

- **Trials**: 20 optimization trials
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Evaluation Metric**: ROC-AUC

### Ensemble Strategy

The final predictions are generated using a simple averaging ensemble:
```python
ensemble_predictions = (xgboost_probabilities + lightgbm_probabilities) / 2
```

This approach combines the strengths of both models and typically provides more robust predictions than either model alone.

## Results

### XGBoost Performance
- **Best Cross-Validation ROC-AUC**: 0.8330
- **Optimal Parameters**:
  - n_estimators: 800
  - max_depth: 7
  - learning_rate: 0.0098
  - subsample: 0.579
  - colsample_bytree: 0.616
  - gamma: 4.214
  - min_child_weight: 10
  - scale_pos_weight: 7.539

### LightGBM Performance
- **Best Cross-Validation ROC-AUC**: 0.8244
- **Optimal Parameters**:
  - n_estimators: 240
  - max_depth: 3
  - learning_rate: 0.053
  - num_leaves: 25
  - min_child_samples: 48
  - scale_pos_weight: 5.089

### Key Insights
- XGBoost achieved slightly better performance (0.8330) compared to LightGBM (0.8244)
- Both models effectively handled the class imbalance through optimized `scale_pos_weight`
- The ensemble approach leverages the complementary strengths of both models
- Stratified K-Fold cross-validation ensured reliable performance estimates

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/themreyildirim/attrition-classification-with-optuna.git
cd attrition-classification-with-optuna
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm optuna scipy jupyter
```

## Usage

### Running the Notebook

1. **Extract the dataset**:
```bash
unzip playground-series-s3e3.zip
```

2. **Launch Jupyter Notebook**:
```bash
jupyter notebook xgboost-and-lightgbm-with-optuna.ipynb
```

3. **Execute the cells sequentially** to:
   - Load and explore the data
   - Perform feature engineering and preprocessing
   - Optimize model hyperparameters using Optuna
   - Train final models and generate predictions
   - Create submission file

### Expected Outputs

The notebook will generate:
- **Visualizations**: Distribution plots for numerical features
- **Optimization logs**: Optuna trial progress and best parameters
- **Submission file**: `submission_xgboost_lgbm_ensemble.csv` with predictions for the test set

### Customization

You can adjust the following parameters in the notebook:
- `N_TRIALS`: Number of Optuna optimization trials (default: 20)
- `N_SPLITS`: Number of cross-validation folds (default: 5)
- `THRESHOLD`: Classification threshold for converting probabilities to classes (default: 0.5)

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SciPy
- Jupyter Notebook

### Detailed Requirements
```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
xgboost>=1.4.0
lightgbm>=3.2.0
optuna>=2.10.0
scipy>=1.5.0
jupyter>=1.0.0
```

## Project Structure

```
attrition-classification-with-optuna/
├── xgboost-and-lightgbm-with-optuna.ipynb  # Main analysis notebook
├── playground-series-s3e3.zip               # Dataset archive
├── train.csv                                 # Training data (extracted)
├── test.csv                                  # Test data (extracted)
├── sample_submission.csv                     # Sample submission format
├── README.md                                 # This file
├── LICENSE                                   # MIT License
└── .gitignore                                # Git ignore rules
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Emre Yıldırım**
- GitHub: [@themreyildirim](https://github.com/themreyildirim)

## Acknowledgments

- Dataset provided by Kaggle Playground Series
- Optuna framework for hyperparameter optimization
- XGBoost and LightGBM communities for excellent gradient boosting libraries

---

**Note**: This project is for educational and competition purposes. The techniques demonstrated here can be adapted for real-world employee attrition prediction systems.
