# to execute it, run the following command in the terminal:
# streamlit run app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# showing the title
st.title('Breast Cancer Predictor')

# loading the scaler and the model saved
scaler = joblib.load('.\saved_models\scaler.pkl')
model  = joblib.load('.\saved_models\model.pkl')

# loading the data that the model needs, 
# using the range identified in data.DESCR attribute
st.write('')
st.write('')
mean_radius          = st.slider('Mean Radius',            6.98,   28.11)
mean_perimeter       = st.slider('Mean Perimeter',        43.79,  188.50)
mean_area            = st.slider('Mean Area',            143.50, 2501.00)
mean_concavity       = st.slider('Mean Concavity',         0.00,    0.43)
mean_concave_points  = st.slider('Mean Concave Points',    0.00,    0.21)
st.write('')
st.write('')
worst_radius         = st.slider('Worst Radius',           7.93,   36.04)
worst_perimeter      = st.slider('Worst Perimeter',       50.41,  251.20)
worst_area           = st.slider('Worst Area',           185.20, 4254.00)
worst_concavity      = st.slider('Worst Concavity',        0.00,    1.26)
worst_concave_points = st.slider('Worst Concave Points',   0.00,    0.30)

# creating a dataframe with the data, 
# and filling the other (not necesary) columns with average values
data = {
    'mean radius':              [mean_radius],
    'mean texture':             20.0,
    'mean perimeter':           [mean_perimeter],
    'mean area':                [mean_area],
    'mean smoothness':          0.1,
    'mean compactness':         0.2,
    'mean concavity':           [mean_concavity],
    'mean concave points':      [mean_concave_points],
    'mean symmetry':            0.2,
    'mean fractal dimension':   0.07,
    'radius error':             1.5,
    'texture error':            2.0,
    'perimeter error':          10.0,
    'area error':               300.0,
    'smoothness error':         0.01,
    'compactness error':        0.1,
    'concavity error':          0.1,
    'concave points error':     0.02,
    'symmetry error':           0.03,
    'fractal dimension error':  0.01,
    'worst radius':             [worst_radius],
    'worst texture':            30.0,
    'worst perimeter':          [worst_perimeter],
    'worst area':               [worst_area],
    'worst smoothness':         0.1,
    'worst compactness':        0.5,
    'worst concavity':          [worst_concavity],
    'worst concave points':     [worst_concave_points],
    'worst symmetry':           0.4,
    'worst fractal dimension':  0.1
}
df = pd.DataFrame(data)

# scaling the data
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# chosing the columns that need the model
df        = df       [['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points', 'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']]
df_scaled = df_scaled[['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points', 'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']]

# joining the dataframes with original and scaled data 
df_to_show = pd.concat([df, pd.DataFrame(df_scaled, columns=df.columns)], axis=0) 

# showing the data
st.write('')
st.write('')
st.write('Original Data (1st row) and Scaled Data (2nd row)')
st.write(df_to_show)

# predicting the class
prediction = model.predict(df_scaled)

# showing the prediction
st.write('')
st.write('')
prediction = '1 (MALIGNANT)' if prediction == 1 else '0 (BENIGN)'
st.write('The predicted class is: ', prediction)
