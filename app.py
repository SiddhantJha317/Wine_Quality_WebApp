#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

#app heading
st.write("""
# My Wine Quality Prediction App
This app predicts the ***Wine Quality*** that's it btw
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.0, 16.0, 12.6)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.10,1.6 , 0.34)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.34 , 0.04)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,280.0 , 31.4)
        alcohol=st.sidebar.slider('alcohol', 8.42,15.0, 8.9)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input')
st.write(df)
#reading csv file
data=pd.read_csv("winequality-red.csv")
X =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(data['quality'])
#random forest model
pipeline=make_pipeline(preprocessing.StandardScaler(),
                        RandomForestClassifier(n_estimators=100))
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestclassifier__max_depth' : [None, 10, 7, 5, 3, 1]}                        
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X, Y)
st.subheader('Wine quality labels')
st.write(pd.DataFrame({
   'wine quality': [3, 4, 5, 6, 7, 8 ]}))

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)