import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load data
dir_path = r"C:\Users\gtregoat\workspace\deep_learning\Kaggle\Titanic"
df_train = pd.read_csv(dir_path + r"\train.csv")
df_test = pd.read_csv(dir_path + r"\test.csv")
labels = pd.read_csv(dir_path + r"\gender_submission.csv")
df_test = df_test.merge(labels, on=["PassengerId"])

# List of variables to use
categorical_variables = ["Pclass", "Sex", "Embarked"]
numeric_variables = ["Age", "SibSp", "Parch", "Fare"]
total_list = categorical_variables + numeric_variables + ["Survived"]
df_train = df_train[total_list]
df_test = df_test[total_list]

# Change to good category type
def to_good_type(df, categorical_colums, numeric_columns):
    for col in categorical_colums:
        df[col] = df[col].astype('category')
    for col in numeric_columns:
        df[col] = df[col].astype('float')
    return df
df_train = to_good_type(df_train, categorical_variables, numeric_variables)
df_test = to_good_type(df_test, categorical_variables, numeric_variables)

# To categorical
df_train = pd.get_dummies(df_train, dummy_na=True)
df_test = pd.get_dummies(df_test, dummy_na=True)

# Deal with NaN for numeric variables
for col in numeric_variables:
    df_train[col] = df_train[col].fillna(df_train[col].mean())
    df_test[col] = df_test[col].fillna(df_test[col].mean())

# Minmax numeric columns
scaler = MinMaxScaler(feature_range=(0, 1))
df_train[numeric_variables] = scaler.fit_transform(df_train[numeric_variables])
df_test[numeric_variables] = scaler.fit_transform(df_test[numeric_variables])

# Build a model
# xgboost
from xgboost import XGBClassifier
# fit model no training data
model = XGBClassifier()
data_columns = [i for i in df_train.columns if i not in ["Survived"]]
model.fit(df_train[data_columns].values, df_train["Survived"].values)
# make predictions for test data
y_pred = model.predict(df_test[data_columns].values)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(df_test["Survived"].values, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(df_test["Survived"].values, predictions))
