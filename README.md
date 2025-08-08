# 💨 Wind Power Forecasting — Linear Regression & SVM

This repository predicts wind turbine power output using Linear Regression and Support Vector Regression (SVR), trained on SCADA data from wind turbines.   
The dataset is loaded from multiple .csv files, combined into a single DataFrame, and enriched with year/month/day/hour features derived from timestamps

---
## 📂 Project Structure
- LR_AllData.py            # Linear Regression model training & evaluation
- SVM_AllData.py           # Support Vector Regression model training & evaluation
- README.md                # Project documentation
---
## 🧠 Methods
**Feature Engineering**
Converts Time to datetime and extracts year, month, day, hour columns.
Adds region column to distinguish datasets from different files.  
**Target Variable:** Power  
**Data Split:** 80% train / 20% test  
**Scaling:** Linear Regression: StandardScaler (features only)
**SVR:** StandardScaler (features and target, target is inverse-transformed for evaluation)
**Metrics:** R² and MSE  
**Visualization:** First 300 test samples — actual vs. predicted plots

---
## 🔧 Requirements
- Python 3.9+ (3.10 recommended)  
- NumPy, Pandas, scikit-learn, Matplotlib

---
## 📈 Expected Output

**Linear Regression:**  
Console: MSE & R² values  
Graph: Actual vs. Predicted (first 300 test points)  

**SVR:**  
Console: R² & MSE  
Graph: Actual vs. Predicted (first 300 test points)  

---
## 📝 Data Requirements

Each CSV must contain:   
- Time — timestamp in a format readable by pandas.to_datetime  
- Power — numeric  
- Other columns should be numeric or convertible to numeric. 

---
## 📊 Data Source
The dataset used in this project was obtained from Kaggle:  
[Wind Power Generation Data - Forecasting](https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting)

---

## 📜 License
You can freely use and modify this project.
