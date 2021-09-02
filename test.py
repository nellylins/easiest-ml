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

# selecting target
y = st.selectbox(
   'What column is the target?',
    cols)
# selecting features
x = st.multiselect(
    'What columns would you like to use in the model?',
    cols
)

#Loading in all models
allModels = ["Nearest Neighbors", "Decision Tree Classifier", "Random Forest Classifier", "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
allMods = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        LinearRegression(),
        DecisionTreeRegressor(random_state=0),
        RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0)
           ]
#Creating instance
modType = 'classification'

# selecting type of model pt 1 (selecting to select)
modChoose = st.selectbox(
    "Would you like to select which model type to use?",
    ("Choose One", "Yes", "No")
)
#Selection options
if modChoose == "Choose One":
    st.write("")
elif modChoose == "Yes":
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

    # Creating dataframes for x & y values
    X = dfTrain[x]
    Y = dfTrain[y]
    # Splitting training & testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=0)
    ##### If user does not want to choose their specific model type
    if modChoose == "No":
        bestScore = 0
        ## User asked what type of data it is, chose classification
        if modType == 'classification':
            # test for best model
            for name, clf in zip(clf_names, classifiers):
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                #Determine best based on score
                if score > bestScore:
                    bestScore = score
                    bestClf = name
                    bestClass = clf
            # print best accuracy
            st.write('You are using', listToString(x), 'to predict', y, 'with ', bestClf, "and a training accuracy of ", bestScore, ". The model's predictions are below")
            return bestClass
        ## User asked what type of data it is, chose regression
        elif modType == 'regression':
            # test different models, such as...
            for name, reg in zip(reg_names, regressors):
                reg.fit(X_train, y_train)
                score = reg.score(X_test, y_test)
                # Determine best based on score
                if score > bestScore:
                    bestScore = score
                    bestRg = name
                    bestReg = reg
            # print best accuracy
            st.write('You are using', listToString(x), 'to predict', y, 'with ', bestRg, "and a training R squared of ", bestScore, ". The model's predictions are below")
            return bestReg
    ######## If user does want to select their own model
    else:
        i = allModels.index(modType)
        chosen = allMods[i]
        chosen.fit(X_train, y_train)
        score = chosen.score(X_test, y_test)
        st.write('You are using', listToString(x), 'to predict', y, 'with ', modType, "and a training accuracy/R squared of ",
                 score, ". The model's predictions are below")
        return chosen


########Model testing setup


#What type of output
predictType = st.selectbox(
    "Would you like to batch predict or use a sample?",
    ("Choose One", "Batch", "Sample")
)


dfTest = pd.DataFrame()

# Choose model button
if predictType == "Choose One":
    st.write("Select batch or Sample")
#Batch predict
elif predictType == "Batch":
    # upload batch testing file
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
# Sample predict
else:
    index = 0
    # entering values for features
    for i in x:
        dfTest.loc[0, i] = st.text_input("Enter " + i)
    #Make prediction
    if st.button('Predict using model'):
        fitModel = train_model()
        predict = fitModel.predict(dfTest[x])
        dfTest = dfTest.assign(prediction=predict)
        st.write(dfTest)