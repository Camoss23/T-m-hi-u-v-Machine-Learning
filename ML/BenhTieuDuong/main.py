from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
import joblib
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / 'diabetes.csv'
data = pd.read_csv(file_path)
target = data['Outcome']
data = data.drop('Outcome', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
num_pipeline = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
compose = ColumnTransformer(transformers=[
    ('num',pipeline,num_pipeline),
    ])
model = Pipeline(([
    ('preprocessor', compose),
    ('Random' , RandomForestClassifier(n_estimators=100, random_state=42 )),
]))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
joblib.dump(model, "model.pkl")
