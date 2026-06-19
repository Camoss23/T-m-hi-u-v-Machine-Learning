import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error
from lazypredict.Supervised import Lazy
import joblib
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / 'StudentScore.xls'
data = pd.read_csv(file_path)
#print(data.head())
#print(data.describe())
#print(data.info())
#ProfileReport = ProfileReport(data, title="Student Score")
#ProfileReport.to_file(BASE_DIR / 'StudentScore.html')
target = 'math score'
x = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
feature_numbers =  ['reading score','writing score']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
                                  ])
onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False))
])
education_rank = ["some high school",
                   "high school",
                   "some college",
                    "associate's degree",
                   "bachelor's degree",
                   "master's degree",
                  ]
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('odi',OrdinalEncoder(categories=[education_rank]))
])
preprocessor = ColumnTransformer(transformers=[
    ('num',num_transformer,feature_numbers),
    ('nom_features',onehot_transformer,["race/ethnicity",'gender','lunch','test preparation course']),
    ('odi_features',ordinal_transformer,['parental level of education'])
    ])
model = Pipeline(steps=[
    ("Preprocessor", preprocessor),
    #("Liner", LinearRegression()),
    ("Regressor", RandomForestRegressor(random_state=42)),
])
params = {
    "Regressor__n_estimators": [50, 100, 200],
    "Regressor__criterion": ["squared_error", "absolute_error", "friedman_mse"],
    #"max_depth" : [None,2,5]
}
gs_model = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring="r2",
    cv=5,
    verbose=2,
    n_jobs= 5
)
gs_model.fit(X_train,y_train)
print(gs_model.best_params_)
print(gs_model.best_score_)
#rs =  model.fit(X_train, y_train)
#y_pared = model.predict(X_test)
#print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pared)}")
#print(f"Mean Squared Error: {mean_squared_error(y_test, y_pared)}")
#print(f"R2 Score: {r2_score(y_test, y_pared)}")
#print(f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pared)}")
#reg = LazyRegressor(verbose= 0,ignore_warnings=True,custom_metric=None)
#models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#print(models)
#joblib.dump(model,"model.pkl")
#joblib.dump(models,"models.pkl")
#print(model['Preprocessor'].score(X_test, y_test))
#print(model['Liner'].score(X_test, y_test))
#test_sample = pd.DataFrame([{
#   "reading score": 85,
#    "writing score": 88,
#    "race/ethnicity": "group C",
#   "gender": "female",
#    "lunch": "standard",
#    "test preparation course": "completed",
#    "parental level of education": "bachelor's degree"
#}])
#predicted_score = model.predict(test_sample)

#print(f"Predicted Math Score for the test student: {predicted_score[0]:.2f}")