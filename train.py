import pandas as pd
import joblib
from pipeline import build_pipelines
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


Data = pd.read_csv(r"C:\Users\naman\OneDrive\Desktop\autoML\healthcare-dataset-stroke-data.csv")

X = Data.drop(["id", "stroke"], axis=1) 
y = Data["stroke"]

categorical = X.select_dtypes(include="object").columns.tolist()
numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

pipeline = build_pipelines(categorical, numerical)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(" Classification Report:\n", classification_report(y_test, y_pred))
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


joblib.dump(pipeline,"model.pkl")