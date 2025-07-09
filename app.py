import numpy as np
import streamlit as st 
import joblib
scaler = joblib.load('sc.pkl')
model = joblib.load('model.pkl')

st.title("Churn Prediction App")
st.divider()
st.write("Please enter the values and hit the prediction button for getting a prediction")
st.divider()
age = st.number_input("Enter the age",min_value = 10, max_value=100,value=30)
tenure = st.number_input("Enter tenure",min_value=0,max_value=130, value = 10)
monthlycharge = st.number_input("Enter the monthly charge",min_value=30,max_value=150)
gender = st.selectbox("Enter the gender",['Male','Female'])
st.divider()
btn = st.button("Predict")
st.divider()
if btn:
    gender_selected = 1 if gender=='Male' else 0
    x = [age,gender_selected,tenure,monthlycharge]
    X1 = np.array(x)
    x_array = scaler.transform([X1])
    prediction = model.predict(x_array)[0]
    predicted = 'Churn' if prediction == 1 else 'Not Churn'
    st.balloons()
    st.write(f'Predicted:{predicted}')
else:
    st.write("Please enter the value and use predict button")
