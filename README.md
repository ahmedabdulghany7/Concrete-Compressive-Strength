# Concrete Compressive Strength Prediction

A comprehensive machine learning project to predict concrete compressive strength using advanced regression algorithms. This project was developed as part of SET 393: Data Mining and Business Intelligence course at Egyptian Chinese University.

## 🔍 Overview

Concrete compressive strength is a critical parameter in structural engineering that determines the safety and durability of construction projects. Traditional testing requires a 28-day curing period, leading to project delays and increased costs. This project leverages machine learning to predict compressive strength based on concrete mixture composition, enabling faster decision-making in construction engineering.

### Key Objectives:
- **Accelerate Quality Control**: Reduce the 28-day waiting period for strength testing
- **Optimize Mix Design**: Help engineers create optimal concrete mixtures
- **Reduce Costs**: Minimize material waste through predictive modeling
- **Enhance Safety**: Ensure structural integrity through accurate predictions

## 📊 Dataset

The dataset contains **1,030 observations** from controlled experiments measuring concrete compressive strength.

### Input Features (kg/m³):
- **Cement**: Primary binding agent (102-540 kg/m³)
- **Blast Furnace Slag**: Industrial byproduct supplement (0-359 kg/m³)
- **Fly Ash**: Coal combustion byproduct (0-200 kg/m³)
- **Water**: Mixing water (121-247 kg/m³)
- **Superplasticizer**: Chemical admixture (0-32 kg/m³)
- **Coarse Aggregate**: Large particles (801-1145 kg/m³)
- **Fine Aggregate**: Small particles/sand (594-992 kg/m³)
- **Age**: Concrete age in days (1-365 days)

### Target Variable:
- **Compressive Strength**: Measured in MPa (2.33-82.6 MPa)

**Data Quality**: Complete dataset with no missing values or duplicates.

## 💻 Implementation

The complete implementation is available in our interactive Kaggle notebook:
**[🔗 View Full Implementation](https://www.kaggle.com/code/ahmedabdulghany/concrete-compressive-strength-dt-xgboost-rf)**

## 📈 Exploratory Data Analysis

### Distribution Analysis
- **Cement**: Normal distribution with slight positive skew (0.509)
- **Water**: Near-normal distribution with minimal skew (0.074)
- **Age**: Strong positive skew (3.264) with high kurtosis (12.104)
- **Blast Furnace Slag & Fly Ash**: Many zero values indicating selective usage
- **Target Variable**: Slight positive skew (0.416), mean ≈ 35.8 MPa

### Key Correlations
- **Cement**: Strongest positive correlation with strength (0.50)
- **Age**: Strong positive correlation (0.33)
- **Water**: Negative correlation (-0.29)
- **Superplasticizer**: Positive correlation (0.31)

### Advanced Analysis
- **Principal Component Analysis**: Applied for dimensionality reduction
- **Multicollinearity Assessment**: No severe multicollinearity detected (acceptable VIF values)
- **Outlier Detection**: IQR method applied; outliers retained as valid mixture formulations

## 🤖 Models

Six regression algorithms were implemented and evaluated:

### 1. **XGBoost Regressor** ⭐ (Best Model)
- Advanced gradient boosting algorithm
- Hyperparameter tuning with grid search
- Cross-validation optimization

### 2. **Random Forest Regressor**
- Ensemble method with 100-300 estimators
- Feature importance analysis
- Robust to outliers

### 3. **Decision Tree Regressor**
- Interpretable single tree model
- Visualization of decision paths
- Baseline ensemble comparison

### 4. **Linear Regression**
- Simple baseline model
- Feature coefficient analysis
- Limited capacity for non-linear relationships

### 5. **K-Nearest Neighbors**
- Instance-based learning (k=5)
- Local pattern recognition
- Distance-based predictions

### 6. **Support Vector Machine**
- RBF kernel implementation
- Non-linear pattern recognition
- Moderate performance

## 📊 Results

### Final Model Performance (After Hyperparameter Tuning)

| Model | MAE | MSE | RMSE | R² Score | CV Score | Training Time |
|-------|-----|-----|------|----------|----------|---------------|
| **XGBoost** ⭐ | **2.91** | **19.82** | **4.45** | **0.92** | **0.92 ± 0.03** | 0.45s |
| Random Forest | 3.76 | 30.43 | 5.52 | 0.88 | 0.88 ± 0.03 | 0.62s |
| Decision Tree | 4.59 | 53.67 | 7.33 | 0.79 | 0.79 ± 0.05 | 0.02s |
| KNN | 6.86 | 74.33 | 8.62 | 0.71 | 0.71 ± 0.04 | 0.03s |
| Linear Regression | 7.75 | 95.98 | 9.80 | 0.63 | 0.63 ± 0.04 | 0.01s |
| SVM | 7.57 | 90.71 | 9.52 | 0.65 | 0.65 ± 0.04 | 0.15s |

### XGBoost Optimal Hyperparameters
- **Learning Rate**: 0.05
- **Max Depth**: 5
- **Min Child Weight**: 3
- **Subsample**: 0.8
- **Column Sample by Tree**: 0.8
- **N Estimators**: 200

### Feature Importance (XGBoost)
1. **Cement**: 28.51% - Most influential component
2. **Age**: 25.16% - Critical for strength development
3. **Water**: 16.43% - Negative impact on strength
4. **Superplasticizer**: 11.25% - Chemical enhancement
5. **Blast Furnace Slag**: 7.62%
6. **Fly Ash**: 5.87%
7. **Fine Aggregate**: 2.83%
8. **Coarse Aggregate**: 2.33%

### Model Validation
- **10-Fold Cross-Validation**: Consistent performance across data splits
- **Learning Curves**: Convergence between training/validation scores
- **Bias-Variance Analysis**: Optimal balance achieved
- **Residual Analysis**: Normal distribution, homoscedasticity confirmed

### Prediction Confidence Intervals
- **90% Prediction Interval**: ±7.3 MPa
- **95% Prediction Interval**: ±8.7 MPa  
- **99% Prediction Interval**: ±11.4 MPa

## 🏗️ Practical Applications

### Early-Age Strength Prediction
- **Timeline Acceleration**: Reduce project timelines by up to 15%
- **Quality Control**: Early identification of potential strength issues
- **Cost Optimization**: Minimize material waste through accurate predictions

### Sustainable Construction
- **Cement Reduction**: 8-12% reduction while maintaining target strength
- **Alternative Materials**: Optimal fly ash (15-25%) and slag (20-30%) replacement rates
- **Carbon Footprint**: Up to 15% reduction in concrete production emissions
- **Water Optimization**: 5-10% water reduction potential

### Decision Support
- **Mix Design Optimization**: Automated mixture composition recommendations  
- **Sensitivity Analysis**: Understanding component variation impacts
- **Regional Adaptation**: Framework for local material customization

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Clone Repository
```bash
git clone https://github.com/ahmedabdulghany7/Concrete-Compressive-Strength.git
cd Concrete-Compressive-Strength
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
```

## 💻 Usage

### Load and Explore Data
```python
# Load dataset
df = pd.read_excel('Concrete_Data.xls')

# Basic information
print(f"Dataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
```

### Make Predictions
```python
# Example prediction with XGBoost model
cement = 540.0
blast_furnace_slag = 0.0
fly_ash = 0.0
water = 162.0
superplasticizer = 2.5
coarse_aggregate = 1040.0
fine_aggregate = 676.0
age = 28

# Prepare input
input_features = np.array([[cement, blast_furnace_slag, fly_ash, water, 
                          superplasticizer, coarse_aggregate, fine_aggregate, age]])

# Scale features (using pre-fitted scaler)
input_scaled = scaler.transform(input_features)

# Predict
predicted_strength = xgb_model.predict(input_scaled)
print(f"Predicted Compressive Strength: {predicted_strength[0]:.2f} MPa")
```

### Model Training Pipeline
```python
# Data preprocessing
X = df.drop('Concrete compressive strength(MPa, megapascals)', axis=1)
y = df['Concrete compressive strength(MPa, megapascals)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost model training
xgb_model = xgb.XGBRegressor(
    learning_rate=0.05, max_depth=5, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8, n_estimators=200, random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = xgb_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f} MPa")
```

## 🔬 Advanced Features

### Cross-Validation Analysis
- 10-fold cross-validation implementation
- Statistical significance testing
- Model stability assessment

### Hyperparameter Optimization
- Grid search with cross-validation
- Automated parameter tuning
- Performance optimization

### Feature Engineering
- Standard scaling normalization
- Feature importance ranking
- Multicollinearity analysis

### Model Interpretability
- SHAP value analysis
- Feature contribution visualization
- Decision tree path explanation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

**Concrete Compressive Strength Prediction – Academic License**

**Copyright © 2025 Ahmed Abdulghany**

**Supervised by:** Assistant Professor Dr. Rasha Saleh  
**Institution:** Department of Software Engineering & Information Technology, Egyptian Chinese University  
**Date:** May 10, 2025

---

### Terms of Use

Permission is hereby granted, free of charge, to any person or institution using this software and related documentation files (the "Software") for **non-commercial academic, research, or educational purposes**, subject to the following conditions:

#### ✅ Permitted Uses
- Academic research and study
- Educational purposes
- Non-commercial analysis and experimentation

#### 📋 Requirements
1. **Attribution** must be given to the original authors and supervisor
2. Any **publications, presentations, or derivative works** that use this Software or data must include proper citation of the original authors and course
3. **Redistribution** of substantial portions must include this license file and the full list of authors and institutional affiliations

#### ❌ Restrictions
- The Software may **not** be used, copied, modified, merged, published, or distributed for **commercial purposes** without prior written permission from the authors or the supervising institution

---

### Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR SUPERVISORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THE SOFTWARE.

---

### Contact

For commercial inquiries or permission requests, please contact:  
📧 **ahmedabdulghany7@gmail.com**

---

## 👨‍💻 Author

**Ahmed Abdulghany**
- **Email**: ahmedabdulghany7@gmail.com
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/ahmedabdulghany/)
- **GitHub**: [GitHub Profile](https://github.com/ahmedabdulghany7)

Supervisor: Assistant Professor Dr. Rasha Saleh

## 🙏 Acknowledgments

- Dr. Rasha Saleh for supervision and guidance
- Egyptian Chinese University for providing the academic framework
- UCI Machine Learning Repository for the original dataset
- The concrete engineering research community for domain insights


⭐ **If you found this project helpful, please give it a star!** ⭐

**Note**: This project demonstrates the application of advanced machine learning techniques to solve real-world engineering problems, showcasing the potential of data-driven approaches in construction and materials science.
