import streamlit as st
import requests

st.title("Job Classification")

title = st.text_input("Title")
description = st.text_input("Description")
industry = st.text_input("Industry")
location = st.text_input("Location")
function = st.text_input("Function")

if st.button("Predict"):
    res = requests.post(
        "http://localhost:8000/predict",
        json={
            "title": title,
            "description": description,
            "industry": industry,
            "location": location,
            "function": function
        }
    )
    st.write("Prediction:", res.json()["prediction"])