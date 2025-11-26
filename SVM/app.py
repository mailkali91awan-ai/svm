import streamlit as st
import pickle
import pandas as pd

st.title("🌸 Iris Flower Prediction App (SVM Kernels)")

# Load pickle models
with open("all_svm_models.pkl", "rb") as f:
    models = pickle.load(f)

st.subheader("Choose SVM Kernel")
kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])

st.subheader("Enter Flower Measurements")

sl = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sw = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
pl = st.number_input("Petal Length", 0.0, 10.0, 1.4)
pw = st.number_input("Petal Width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    df = pd.DataFrame([[sl, sw, pl, pw]],
        columns=["sepal_length","sepal_width","petal_length","petal_width"])

    model = models[kernel]
    prediction = model.predict(df)

    st.success(f"🌼 Predicted Species: **{prediction[0]}**")


