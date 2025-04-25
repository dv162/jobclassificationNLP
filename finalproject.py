import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
import re


def my_func(loc):
    result = re.findall(r" [A-Z]{2}$", loc)
    if len(result) == 1:
        return result[0][1:]
    else:
        return loc


data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data["location"] = data["location"].apply(my_func)  # clear data
# print(len(data["function"].unique())) # check total category
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]  # sort column

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)  # sort row
# ros = SMOTE(random_state=42, k_neighbors=2, sampling_strategy={
#     "director_business_unit_leader": 500, "managing_director_small_medium_company": 500, "specialist": 500
# })  # over simpling to improve performance
x_train["description"] = x_train["description"].fillna("missing")  # replace missing data

# vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
# result = vectorizer.fit_transform(x_train["description"])

## xử lý những cột data
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1), min_df=0.01), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "industry")
])
# pipline steps, use SelectKbest to improve performance model
cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selector", SelectKBest(chi2, k=300)),
    ("classifier", RandomForestClassifier(random_state=42))
])

cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)

print(classification_report(y_test, y_predict))
# print(result.shape)
# print(len(vectorizer.vocabulary_))

#classfition