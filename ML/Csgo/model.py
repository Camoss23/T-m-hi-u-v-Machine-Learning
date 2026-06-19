import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from  sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / 'csgo.csv'
data = pd.read_csv(file_path)
x = data.drop (["team_a_rounds","team_b_rounds","result","day","month","year","date"], axis = 1)
y = data["result"]
le = LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])
onehot_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
compose = ColumnTransformer([
    ("num", num_pipeline,["wait_time_s","match_time_s","ping","kills","assists","deaths","mvps","hs_percent","points"]),
    ("onehot", onehot_pipeline,["map"]),
])
model = Pipeline([
    ("preprocessor", compose),
    ("classifier", RandomForestClassifier(random_state=42,class_weight="balanced")),
])
model.fit(x_train,y_train)
y_pre = model.predict(x_test)
print(model.classes_)
s = model.predict_proba(x_test)
#print(y_pre,classification_report(y_test,y_pre))
for i in range(len(y_pre)):
    print(f"Sample {i}")
    for cls, prob in zip(model.classes_, s[i]):
        print(f"{cls}: {prob*100:.2f}%")
    print("-----")
