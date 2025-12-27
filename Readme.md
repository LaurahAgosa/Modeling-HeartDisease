# MODELING HEART DISEASE

## Project Overview
This project aims to develop a predictive model for heart disease using a dataset containing 14 clinical and demographic attributes, such as age, cholesterol levels, blood pressure, chest pain type, and ECG results. The dataset was sourced from the UCI Machine Learning Repository. Through Exploratory Data Analysis (EDA), patterns and relationships within the data were explored. Multiple machine learning models were trained and evaluated to identify the most reliable model for predicting heart disease risk. Logistic Regression was selected as the final model due to its high accuracy and interpretability.

---

## Features in the Dataset
The dataset contains 14 attributes:

- **age**: Age of the patient  
- **sex**: Gender of the patient (Male/Female)  
- **cp**: Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)  
- **trestbps**: Resting blood pressure (mm Hg)  
- **chol**: Serum cholesterol (mg/dl)  
- **fbs**: Fasting blood sugar > 120 mg/dl (1: true, 0: false)  
- **restecg**: Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: probable/definite left ventricular hypertrophy)  
- **thalach**: Maximum heart rate achieved  
- **exang**: Exercise-induced angina (1: yes, 0: no)  
- **oldpeak**: ST depression induced by exercise relative to rest  
- **slope**: Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)  
- **ca**: Number of major vessels colored by fluoroscopy (0–3)  
- **thal**: Thalassemia (1: fixed defect, 2: normal, 3: reversible defect)  
- **target**: Presence of heart disease (1: disease, 0: no disease)

---

## Project Objectives

### General Objective
- Analyze and develop a predictive model using clinical and demographic indicators to predict heart disease.

### Specific Objectives
1. Store and query the heart disease dataset using Python.  
2. Clean, explore, and visualize the data.  
3. Identify key risk factors and interpret insights.  
4. Develop and evaluate machine learning models for prediction.  

---

## Data Preprocessing
- Boolean features were converted into categorical or numeric types.  
- Missing values in `thal` and `ca` were replaced with median values.  
- Feature scaling (standardization) was applied for models sensitive to feature ranges.  

---

## Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Distributions of numerical and categorical features were analyzed.  
- **Bivariate Analysis**: Relationships between features and target were visualized with boxplots and countplots.  
- **Multivariate Analysis**: Scatterplots and correlation heatmaps were used to understand interdependencies among features.  

Key insights:
- Heart disease risk is influenced by a combination of demographic and clinical factors.  
- Maximum heart rate tends to decrease with age.  
- No single variable perfectly separates the classes, highlighting the need for multivariate modeling.  

---

## Machine Learning Models

### Models Trained
1. **Logistic Regression**  
2. **Decision Tree**  
3. **Support Vector Machine (SVM)**  
4. **Gradient Boosting (XGBoost)**  

### Model Evaluation
Models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.  

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.885 | 0.90 | 0.88 | 0.89 | 0.94 |
| SVM | 0.787 | 0.85 | 0.72 | 0.78 | 0.93 |
| Decision Tree | 0.770 | 0.79 | 0.78 | 0.78 | 0.76 |
| XGBoost | 0.786 | 0.85 | 0.72 | 0.78 | 0.91 |

**Selected Model**: Logistic Regression, due to highest ROC-AUC and interpretability.  

### Feature Importance (Logistic Regression)
- Features like `cp` (chest pain), `thalach` (max heart rate), `slope`, and `restecg` had the highest positive influence on heart disease prediction.

---

## Predictive System
- Users can input patient data, which is standardized and fed into the model to predict heart disease presence.  
- The system outputs:
  - **0** → No Heart Disease  
  - **1** → Heart Disease  

---

## Tools & Libraries
- Python 3  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- PyODBC, SQLAlchemy (for database connection)  

---

## Conclusion
- Logistic Regression provides a reliable model for predicting heart disease with high accuracy and interpretability.  
- Multivariate analysis is crucial, as heart disease is influenced by a combination of clinical and demographic factors.  
- The predictive system can help identify patients at risk, supporting early intervention and clinical decision-making.  

---

## Future Work
- Incorporate larger and more diverse datasets for improved model generalization.  
- Explore ensemble and deep learning methods for better predictive performance.  
- Deploy the predictive system as a web or mobile application for real-time clinical use.  

---

## Author
Laurah Agosa  



### Feature Importance (Logistic Regression)
- Features like `cp` (chest pain), `thalach` (max heart rate), `slope`, and `restecg` had the highest positive influence on heart disease prediction.
