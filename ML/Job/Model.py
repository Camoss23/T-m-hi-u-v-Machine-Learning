import pandas as pd
import re
import joblib
from sklearn.metrics import classification_report
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
data = pd.read_excel("final_project.ods", dtype="str")
data = data.dropna(axis=0)
def apply_location(i):
            match = re.findall(r"\b[A-Za-z\s.-]+,\s*([A-Z]{2})\b", str(i))
            return match[0] if match else i
data["location"] = data["location"].apply(apply_location)
y = data["career_level"]
x = data = data.drop("career_level",axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer(
    transformers=[
        ("title_tfidf", TfidfVectorizer(), "title"),
        ("desc_tfidf", TfidfVectorizer(min_df=0.1, max_df=0.99), "description",),
        ("industry_tfidf", TfidfVectorizer(), "industry"),
        ("Oho", OneHotEncoder(handle_unknown="ignore"), ["location", "function"])
    ])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])
model = pipeline.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
joblib.dump(model, "model.pkl")

#np.exp, np.log, and np.reshape
#For example, in computer science, an image is represented by a 3D array of shape  (𝑙𝑒𝑛𝑔𝑡ℎ,ℎ𝑒𝑖𝑔ℎ𝑡,𝑑𝑒𝑝𝑡ℎ=3)
# . However, when you read an image as the input of an algorithm you convert it to a vector of shape  (𝑙𝑒𝑛𝑔𝑡ℎ∗ℎ𝑒𝑖𝑔ℎ𝑡∗3,1)
# . In other words, you "unroll", or reshape, the 3D array into a 1D vector.

