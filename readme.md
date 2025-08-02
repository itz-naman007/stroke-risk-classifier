# Stroke Prediction using Automated Machine Learning (AutoML)

This project demonstrates the development of an end-to-end AutoML pipeline for predicting stroke occurrence using structured health-related data. The pipeline automates feature preprocessing, model selection, and ensemble learning using Scikit-learn and XGBoost, without the need for a graphical user interface (GUI).

---

##  Objective

The primary goal of this project is to train a machine learning model capable of identifying individuals at risk of stroke, based on a set of clinical and demographic features. While the model shows promising results during offline evaluation, it is not intended for real-time or clinical decision-making.

---

##  Dataset Overview

The dataset comprises anonymized health records and includes the following attributes:

- **Demographics**: `gender`, `age`, `Residence_type`, `ever_married`, etc.
- **Clinical indicators**: `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`
- **Behavioral aspects**: `smoking_status`, `work_type`
- **Target variable**: `stroke` (binary classification: 0 = No stroke, 1 = Stroke)

The target distribution is highly imbalanced, which is reflected in model performance, particularly in recall for the minority class.
Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

---

##  Technologies and Libraries Used

- **Python 3.12**
- `pandas` for data manipulation
- `scikit-learn` for modeling, pipelines, and evaluation
- `xgboost` for gradient boosting integration
- `joblib` / `pickle` for model serialization

---

##  Pipeline Architecture

The `build_pipeline()` function creates a unified pipeline that performs the following steps:

1. **Categorical Encoding** using `OneHotEncoder`
2. **Numerical Scaling** using `StandardScaler`
3. **Model Stacking**:
    - Base learners: `RandomForestClassifier`, `XGBoostClassifier`
    - Meta learner: `LogisticRegression`
4. **Feature Selection** using `SelectFromModel` (based on feature importance)
5. **Final Classification** through a `StackingClassifier`

---

##  Model Performance

The model achieved the following metrics on the test set:

- **Accuracy**: `96.28%`
- **Precision (Class 1)**: `33%`
- **Recall (Class 1)**: `6%`
- **F1-score (Class 1)**: `10%`

### Confusion Matrix:

      Predicted
      0    |    1
    ----------------
0 |   982   |   4
1 |    34   |   2




⚠️ The model exhibits **high precision but low recall for stroke cases (class 1)** due to class imbalance. It struggles to correctly identify true positive stroke cases, which is critical in a healthcare setting.

---

## 💾 Model Export

The trained pipeline can be saved and reused via:

```python
import joblib
joblib.dump(pipeline, 'stroke_model.joblib')

