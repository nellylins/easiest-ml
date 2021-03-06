# import libraries
import os
import pandas as pd
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score
ode = OrdinalEncoder()

# Random algos
def listToString(s):
    str1 = ", "
    return (str1.join(s))

# Header
st.write("""
# Build your model
""")

# clean file
def clean_dataset(df):
    #Dropping NAs
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    #Fixing objects (strings) - turning to encoded
    obj_df = df.select_dtypes(include=['object']).copy()
    objcols = obj_df.columns.tolist()
    for col in objcols:
        df[col] = ode.fit_transform(obj_df[[col]])
    return df[indices_to_keep]

# upload training file
dfTrain = pd.DataFrame()
cols = []
uploaded_file = st.file_uploader("Choose a csv file to train the model with")
if uploaded_file is not None:
  dfTrain = pd.read_csv(uploaded_file)
  dfTrain = clean_dataset(dfTrain)
  st.write(dfTrain)
  cols = dfTrain.columns.tolist()

# selecting target, features, & type of model
y = st.selectbox(
   'What column is the target?',
    cols)
x = st.multiselect(
    'What columns would you like to use in the model?',
    cols
)
#modType = st.selectbox(
#    'Is this classification or regression?',
#    ('classification', 'regression')
#)
allModels = ["Nearest Neighbors", "Decision Tree Classifier", "Random Forest Classifier", "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
allMods = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        LinearRegression(),
        DecisionTreeRegressor(random_state=0),
        RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0)
           ]
modType = 'classification'

modChoose = st.selectbox(
    "Would you like to select which model type to use?",
    ("Yes", "No")
)
if modChoose == "Yes":
    # Model options
    modType = st.selectbox(
        "Which model would you like to use?",
        allModels
    )
else:
    modType = st.selectbox(
        'Is this classification or regression?',
        ('classification', 'regression')
     )


# Train model
def train_model():
    # Bring in models (classification)
    clf_names = ["Nearest Neighbors", "Decision Tree", "Random Forest"]
    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]
    # Bring in models (regression)
    reg_names = ["Linear Regression", "Decision Tree", "Random Forest"]
    regressors = [
        LinearRegression(),
        DecisionTreeRegressor(random_state=0),
        RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0)
    ]

    X = dfTrain[x]
    Y = dfTrain[y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=0)
    if modChoose == "No":
        bestScore = 0
        if modType == 'classification':
            # test different models
            for name, clf in zip(clf_names, classifiers):
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                if score > bestScore:
                    bestScore = score
                    bestClf = name
                    bestClass = clf

            # pick best accuracy
            st.write('You are using', listToString(x), 'to predict', y, 'with ', bestClf, "and a training accuracy of ", bestScore, ". The model's predictions are below")
            return bestClass
        elif modType == 'regression':
            # test different models
            for name, reg in zip(reg_names, regressors):
                reg.fit(X_train, y_train)
                score = reg.score(X_test, y_test)
                if score > bestScore:
                    bestScore = score
                    bestRg = name
                    bestReg = reg

            # pick best accuracy
            st.write('You are using', listToString(x), 'to predict', y, 'with ', bestRg, "and a training R squared of ", bestScore, ". The model's predictions are below")
            return bestReg
    else:
        i = allModels.index(modType)
        chosen = allMods[i]
        chosen.fit(X_train, y_train)
        score = chosen.score(X_test, y_test)
        st.write('You are using', listToString(x), 'to predict', y, 'with ', modType, "and a training accuracy/R squared of ",
                 score, ". The model's predictions are below")
        return chosen



#predictType = st.selectbox(
#    "Would you like to batch predict or use a sample?",
#    ("Batch", "Sample")
#)


# upload testing file
dfTest = pd.DataFrame()
uploaded_file = st.file_uploader("Choose a csv file to predict on")
if uploaded_file is not None:
  dfTest = pd.read_csv(uploaded_file)
  dfTest = clean_dataset(dfTest)
  #st.write(dfTest)
  cols = dfTest.columns.tolist()


# Make predictions
if st.button('Test Model'):
    fitModel = train_model()
    predict = fitModel.predict(dfTest[x])
    dfTest = dfTest.assign(prediction=predict)
    st.write(dfTest)
