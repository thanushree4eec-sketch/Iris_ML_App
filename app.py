
import streamlit as st
import joblib
import numpy as np

st.title("Iris Flower Prediction")
st.write("Enter the flower measurements below.")

model = joblib.load("iris_model.pkl")

sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):
    values = np.array([[
        sepal_length, sepal_width,
        petal_length, petal_width
    ]])

    prediction = model.predict(values)[0]
    st.success("Predicted Species: " + prediction)
