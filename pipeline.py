from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import pandas as pd


def  build_pipelines(categorical_f,numeriacl_f):  #here categorical_f ,eans categorical features and numerical_f means numerical features
# numerical pipeline for filling missing values and standardization
    numerical_pipeline = Pipeline([("imputer",SimpleImputer(strategy="mean")),("scaler",StandardScaler())])
# categorical pipeline for imputes missing values with most frequent category and one hot encoding and prevent crashes if test data is unseen
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])
# combine preprocessing  which is applied on numerical pipeline and categorical pipeline and column tranformers is used for keep everything alligned.
    preprocess = ColumnTransformer([("num",numerical_pipeline,numeriacl_f),("cat",categorical_pipeline,categorical_f)])
# feature selection, here it trains a randomforest classifier to randk feature importance,select those features whose importance is above the median and help remove noises
    feature_selection = SelectFromModel(RandomForestClassifier(n_estimators=100),threshold="median")
# model stacking , here i am using randomforest and XGboost as base model because they are tree based and ensembled and LogisticRegression as final because learns to combine base mdoel output
    base_model = [("rf",RandomForestClassifier(n_estimators=100,random_state=30)),("xgb",XGBClassifier(use_label_encoder=False,eval_metric='logloss'))]
    final_model = LogisticRegression()
    stack_model=StackingClassifier(estimators=base_model,final_estimator=final_model)
#  full pipeline
    pipeline = Pipeline([("preprocesser",preprocess),("feature_select",feature_selection),("stacked_model",stack_model)])


    return pipeline