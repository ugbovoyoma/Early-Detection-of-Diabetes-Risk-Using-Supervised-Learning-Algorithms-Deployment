# Early Detection of Diabetes Risk Using Supervised Learning Algorithms

## Project Objective

The primary objective of this project is to develop and deploy a machine learning model for the early prediction of diabetes based on key physiological and demographic features. These include:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI (Body Mass Index)  
- Diabetes Pedigree Function  
- Age  

By analysing these features, the model aims to accurately identify individuals at risk of developing diabetes, supporting timely medical intervention and improved patient outcomes.

Additionally, the model is designed for real-world deployment as a user-friendly application or web service to assist healthcare professionals, researchers, and patients in making informed, data-driven decisions. The Deployed Model can be used here: https://diabetesdetection247.streamlit.app/

---

## Dataset

- **Source**: Pima Indians Diabetes Dataset  
- **Rows**: 768  
- **Columns**: 9  
- **Target Variable**: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)  
- **Missing Values**: None  
- **Imbalance**: Yes (More non-diabetic cases)
- **Reference:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
---

## Tools and Libraries

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn (Logistic Regression, SVM, Decision Tree)
- Imbalanced-learn (SMOTE for class balancing)
- Evaluation Metrics (Accuracy, Precision, Recall, ROC AUC)

---

## Workflow

### 1. Data Preprocessing
- Load CSV using pandas
- Basic EDA and statistical summary
- Feature standardisation using `StandardScaler`
- Resampling using `SMOTE` to balance class distribution

### 2. Data Splitting
- `train_test_split()` with 80/20 ratio
- Standardised input features and balanced target labels

### 3. Model Training and Evaluation

#### Logistic Regression
- **Accuracy**: ~0.75  
- **Precision**: ~0.74  
- **Recall**: ~0.77  
- **ROC AUC**: ~0.75  

#### Support Vector Machine (RBF Kernel)
- **Accuracy**: ~0.80  
- **Precision**: ~0.76  
- **Recall**: ~0.88  
- **ROC AUC**: ~0.80  

#### Decision Tree
- **Accuracy**: ~0.72  
- **Precision**: ~0.72  
- **Recall**: ~0.72  
- **ROC AUC**: ~0.71  

### 4. Visualisations
- Pairplot of features
- Feature correlation heatmap
- Outcome distribution barplot
- Confusion matrices
- ROC curves for classification performance

---

## Key Findings

- SVM showed the best performance in terms of recall and AUC.
- SMOTE helped in balancing the dataset, improving model fairness.
- Logistic Regression and Decision Tree also yielded reasonable predictive power.

---

## Future Work

- Add more advanced ensemble methods (e.g., XGBoost, AdaBoost)
- Hyperparameter tuning using `GridSearchCV`
- Incorporate additional health indicators for better prediction accuracy

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/ugbovoyoma/diabetes-risk-prediction.git
   cd diabetes-risk-prediction

## Author's Details
Author: Ugbovo Yoma

Email: ugbovoyoma@gmail.com

Linkedin: https://www.linkedin.com/in/ugbovoyoma/   
