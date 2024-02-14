import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_data():
  data = pd.read_csv("data/data.csv")

  return data

def sidebar():
  st.sidebar.header("Diabetes Measurements")

  data = get_data()

  slider_labels=[
        ("Number Of Pregnancies","Pregnancies"),
        ("Glucose in Blood", "Glucose"),
        ("Blood Pressure (mm Hg)", "BloodPressure"),
        ("Skin Thickness (mm)", "SkinThickness"),
        ("Insulin (mu U/ml)", "Insulin"),
        ("BMI (weight in kg/(height in m)^2)", "BMI"),
        ("Diabetes pedigree function", "DiabetesPedigreeFunction"),
        ("Age (years)", "Age"),
    ]
  
  input = {}
   
  for label, key in slider_labels:
    input[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      step = 0.5

    )

  return input

def predictions(input_data):
  model = pickle.load(open("model/model-1.pkl", "rb"))

  test_data = np.array(list(input_data.values())).reshape(1,-1)

  predict = model.predict(test_data)

  st.subheader("Diabetes prediction")
  st.write("The Result is:")
  if predict[0] == 0:
    st.write("<span class='diagnosis no_diabetes'>You're Good 🥰</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis diabetes'>Better Check To Doctor 🤒</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of No Diabetes : ", model.predict_proba(test_data)[0][0])
  st.write("Probability of being Diabetes: ", model.predict_proba(test_data)[0][1])
  
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def main():
  st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  input_data = sidebar()
  
  with st.container():
    st.markdown("### Hi I'm [Rifky](https://linktr.ee/rifkyuntara), Welcome to My App Predictor 💻")
    st.image("assets/image.jpg")
  col1, col2= st.columns([4,2])

  with col1:
    predictions(input_data)
  with col2:
    st.write("#### Notes")
    st.write("This prediction App is specifically for women, because the historical data used is female patient data. However, if men want to try it you can set pregnancies to '0'")

if __name__ == '__main__':
  main()
