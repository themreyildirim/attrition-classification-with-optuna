# HR Analytics: Employee Attrition Classification with Optuna

Predicting employee turnover using XGBoost and LightGBM optimized with Optuna hyperparameter tuning.

## 📊 Project Overview

This project implements a complete machine learning pipeline for predicting employee attrition using advanced gradient boosting models. The pipeline includes:

1. **Exploratory Data Analysis (EDA)** - Comprehensive analysis of HR data with visualizations
2. **Hyperparameter Tuning** - Automated optimization using Optuna framework
3. **Model Evaluation** - Performance comparison and detailed metrics

## 🚀 Features

- **Data Generation**: Synthetic HR dataset with realistic attrition patterns
- **Exploratory Analysis**: Statistical analysis and visualizations of key factors
- **Model Optimization**: Automated hyperparameter tuning for both XGBoost and LightGBM
- **Comprehensive Evaluation**: Multiple metrics and confusion matrices
- **Feature Importance**: Analysis of key factors contributing to attrition

## 📋 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Dependencies

- pandas==2.1.4
- numpy==1.26.2
- scikit-learn==1.3.2
- xgboost==2.0.3
- lightgbm==4.1.0
- optuna==3.5.0
- matplotlib==3.8.2
- seaborn==0.13.0
- jupyter==1.0.0

## 🔧 Usage

### 1. Generate Dataset

Generate a synthetic HR analytics dataset:

```bash
python generate_data.py
```

This creates `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` with 1,470 employee records.

**Note:** Alternatively, you can use `prepare_data.py` if you have access to the actual IBM HR Analytics dataset from Kaggle. Place the CSV file in the `data/` directory and the script will validate it.

### 2. Exploratory Data Analysis

Run the EDA script to analyze the data and generate visualizations:

```bash
python exploratory_data_analysis.py
```

**Outputs:**
- Statistical summaries of employee data
- Attrition rate analysis by various features
- Visualizations saved in `output/` directory:
  - Attrition distribution
  - Department-wise analysis
  - Overtime impact
  - Age and income distributions
  - Correlation heatmaps
  - Job satisfaction analysis

### 3. Hyperparameter Tuning

Optimize model hyperparameters using Optuna:

```bash
python hyperparameter_tuning.py
```

**Process:**
- Runs 50 trials for XGBoost optimization
- Runs 50 trials for LightGBM optimization
- Uses 5-fold cross-validation with ROC-AUC scoring
- Saves best parameters to `output/best_hyperparameters.json`

### 4. Model Evaluation

Evaluate optimized models on test data:

```bash
python model_evaluation.py
```

**Outputs:**
- Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrices for both models
- Feature importance charts
- Model comparison visualizations
- Results saved to `output/evaluation_results.json`

## 📁 Project Structure

```
attrition-classification-with-optuna/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── output/
│   ├── *.png (visualizations)
│   ├── best_hyperparameters.json
│   └── evaluation_results.json
├── generate_data.py
├── prepare_data.py
├── exploratory_data_analysis.py
├── hyperparameter_tuning.py
├── model_evaluation.py
├── requirements.txt
└── README.md
```

## 📊 Dataset Features

The dataset includes 35 features:

- **Demographics**: Age, Gender, MaritalStatus
- **Work Details**: Department, JobRole, JobLevel
- **Compensation**: MonthlyIncome, HourlyRate, StockOptionLevel
- **Experience**: YearsAtCompany, TotalWorkingYears, YearsInCurrentRole
- **Satisfaction**: JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance
- **Performance**: PerformanceRating, PercentSalaryHike
- **Work Patterns**: OverTime, BusinessTravel, DistanceFromHome
- **Target**: Attrition (Yes/No)

## 🎯 Key Findings

The analysis reveals several factors associated with higher attrition:

1. **Overtime**: Employees working overtime show significantly higher attrition
2. **Job Satisfaction**: Lower satisfaction correlates with increased turnover
3. **Years at Company**: Newer employees (< 2 years) have higher attrition rates
4. **Work-Life Balance**: Poor work-life balance increases attrition risk
5. **Distance**: Employees living farther from work show higher attrition

## 🏆 Model Performance

Both XGBoost and LightGBM models are optimized and evaluated on:

- **Accuracy**: Overall prediction correctness
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

Results are compared side-by-side with visualizations showing relative performance.

## 🔍 Feature Importance

The models identify the most influential features for predicting attrition:

- Monthly Income
- Overtime status
- Years at Company
- Age
- Job Satisfaction
- Distance from Home
- Environment Satisfaction
- Total Working Years

## 📈 Visualization Outputs

The pipeline generates the following visualizations:

1. Attrition distribution
2. Attrition by department
3. Attrition by overtime
4. Age distribution by attrition
5. Monthly income by attrition
6. Years at company analysis
7. Correlation heatmap
8. Job satisfaction impact
9. Model comparison chart
10. Confusion matrices (XGBoost & LightGBM)
11. Feature importance charts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Created by [themreyildirim](https://github.com/themreyildirim)

## 🙏 Acknowledgments

- IBM HR Analytics Employee Attrition dataset (structure inspiration)
- Optuna framework for hyperparameter optimization
- XGBoost and LightGBM teams for excellent gradient boosting implementations
