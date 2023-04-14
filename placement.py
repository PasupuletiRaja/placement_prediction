import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Create a dictionary for stream and their respective values
stream_dict = {'Electronics And Communication':3,
               'Computer Science':1,
               'Information Technology':4,
               'Mechanical':5,
               'Electrical':6,
               'Civil':2}

# Create a function to transform the input data
def transform_input_data(gender, stream, internships, cgpa, hostel, backlog):
    stream_value = stream_dict[stream]
    gender_value = 0 if gender == 'Male' else 1
    hostel_value = 1 if hostel == 'Yes' else 0
    backlog_value = 1 if backlog == 'Yes' else 0
    transformed_data = np.array([[gender_value, stream_value, internships, cgpa, hostel_value, backlog_value]])
    transformed_data = scaler.transform(transformed_data)
    return transformed_data

# Create a function to predict the placement probability
def predict_placement_probability(transformed_data):
    placement_probability = model.predict_proba(transformed_data)[:,1]
    return placement_probability[0]

# Create the Streamlit app
st.title('Placement Prediction')

# Add the input widgets
gender = st.radio('Gender', ['Male', 'Female'])
stream = st.selectbox('Stream', list(stream_dict.keys()))
internships = st.number_input('Internships', value=0)
cgpa = st.number_input('CGPA', value=0.0, format="%.1f")
hostel = st.radio('Hostel', ['Yes', 'No'])
backlog = st.radio('History Of Backlogs', ['Yes', 'No'])

# Add the submit button
if st.button('Submit'):
    transformed_data = transform_input_data(gender, stream, internships, cgpa, hostel, backlog)
    placement_probability = predict_placement_probability(transformed_data)
    st.write(f"Probability of getting placed: {placement_probability:.2%}")
