### python version: 3.10.13

import streamlit as st
import pandas as pd
import numpy as np
import sklearn, pickle, os

st.set_page_config("Diabetes Predictor", page_icon="syringe")
st.title(':syringe: Diabetes Predictor')
st.caption("The model has Accuracy: 80.52, Precision: {1: 0.86, 0: 0.76}, Recall: {1: 0.90, 0: 0.62}, F1 Score: {1: 0.86, 0: 0.68}, RUC-AOC Score: 75.87")
st.divider()


######################## Functions
@st.cache_resource
def load_model():
    """Tries to load the model and returns it. If loading failed then returns 1."""

    model_filename="diabeties-model.pkl"
    dir_path=os.path.dirname(__file__)
    model_file_path=os.path.join(dir_path,model_filename)

    try:
        with open(model_file_path, 'rb') as f:
            model=pickle.load(f)
            print("Model loading successful.")

    except FileNotFoundError:
        print("File not found")
        model=1

    except pickle.UnpicklingError:
        print("Unpickeling model failed.")

    return model

def transform_input(inp):
    """Requires input with reqiured features (list of values), transforms the input using mean and standard deviations used while training, prepared input dataframe and returns it. If anything fails, returns 1"""

    mean_values=np.array([3.83646889, 120.90448625, 68.86541245, 20.74529667,  81.9015919, 32.02026049, 0.47116353, 33.38060781])
    stds=np.array([3.35426829, 32.16270359, 19.81939012, 15.9213503, 118.13950465, 8.03796674, 0.33069381, 11.874336])

    try:
        feats=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        inp=np.array(inp)
        inp = (inp-mean_values) / stds
        print("Input transformed successfully. Preparing input df..")

        inp=pd.DataFrame([inp], columns=feats)
        print("Successful. Returning input df")

        return inp
    except:
        return 1

def predict(model, inp):
    """Takes model and input. Uses the model to get the prediction and returns it."""
    prediction=model.predict(inp)
    return prediction



##################################### Taking input
# ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
col1, col2 = st.columns(2)
with col1:
    Pregnancies=st.number_input("No. of Pregnancies", value=0, format="%i")
    Glucose=st.number_input("Glucose Level", value=150, format="%i")
    BloodPressure=st.number_input("Diastolic (lower)", value=70, format="%i")
    Insulin=st.number_input("Insulin", value=0, format="%i")

with col2:
    Age=st.number_input("Age", value=21, format="%i")
    SkinThickness=st.number_input("Skin Thickness", value=3, format="%i")
    BMI=st.number_input("BMI (Body Mass Index)", value=40.5, format="%f")
    DiabetesPedigreeFunction=st.number_input("Diabeties Predigree Function", value=0.3, format="%f")

st.divider()
submit_btn=st.button("Submit", type="primary", use_container_width=True)

if submit_btn:
    # loading model, transforming input, getting prediction and displaying it.
    with st.spinner("Predicting.."):
        model=load_model()
        if not isinstance(model,int):
            vals=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            # preparing input
            inp=transform_input(vals)

            if not isinstance(inp,int):
                prediction=predict(model,inp)

                if prediction==0:
                    st.write(f":green[Congratulations.] You don't have diabetes.")
                elif prediction==1:
                    st.write(f"You have :red[diabetes.]")
                else:
                    st.error(f"Prediction failed.")

            # if transformation failed
            else:
                print(inp)
                st.error("Input transformation failed.")

        else:
            st.error("Model loading failed.")