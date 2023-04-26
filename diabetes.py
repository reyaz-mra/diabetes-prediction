import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.utils import resample

st.sidebar.image("medcare.png")
st.image("https://encoredocs.com/wp-content/uploads/2019/10/Diabetes-2.jpg")

col1, col2, col3, col4 = st.columns(4)
sex = 0
with col1:
    option = st.selectbox(
        'Gender',
        ('Male', 'Female'))
    if option == 'Male':
        sex = 0
    if option == 'Female':
        sex = 1
with col2:
    Glucose = st.text_input("Glucose")
with col3:
    BloodPressure = st.text_input("Blood Pressure(in mm/Hg)")
    
with col4:
     if sex == 1:
        Pregnancies = st.text_input("No. of Pregnancies")
     else:
        Pregnancies = 0

col1, col2, col3 = st.columns(3)
with col1:
    SkinThickness = st.text_input("Skin Thickness(in mm)")
with col2:
    Insulin = st.text_input("Insulin (in U/ml)")
with col3:
    BMI = st.text_input("Body Mass Index (in Kg/m2)")

col1, col2, col3 = st.columns(3)

with col1:
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
with col2:
    Age = st.text_input("Age")



df = pd.read_csv("diabetes.csv")

# Remove outliers of Glucose
df.drop(df[df['Glucose'] < 25].index, inplace=True)

# Remove Outliers of Blood Pressure
df.drop(df[df['BloodPressure'] < 30].index, inplace=True)

# Remove outliers of SkinThickness
df.drop(df[df['SkinThickness'] > 80].index, inplace=True)

# balancing for dataset
df_majority_0 = df[(df['Outcome']==0)]
df_minority_1 = df[(df['Outcome']==1)]

df_minority_upsampled = resample(df_minority_1,replace=True,n_samples= 477, random_state=42)

df_upsampled = pd.concat([df_minority_upsampled, df_majority_0])

# Splitting dataset for modelling

X = df_upsampled.drop(columns='Outcome')
y = df_upsampled['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=100)

GBCModel = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.3, random_state=0)
GBCModel.fit(X_train, y_train)

if st.button('Diabetes Test Result'):
    result = GBCModel.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    if result[0] == 1:
        st.error("You have diabetes")
    else:
        st.success("You don't have diabetes")



# st.title(GBCModel.score(X_test, y_test)*100)
