import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

st.title('Classification Model Predictor')


st.write("Note")
st.write(f"For gender Enter 1 for male and 0 for female")
st.write("For Location use this list ")
st.write("0---Chicago")
st.write("1---Houston")
st.write("2---Los_Angeles")
st.write("3---Miami")
st.write("4---New York")
# Get input features from the user
input_features = []

# Assuming you have a list of feature names
feature_names = ['Age', 'Gender', 'Location', 'Monthly_Bill','Total_Usage_GB','Subscription_Length_Months']
for feature_name in feature_names:
    value = st.number_input(f"Enter value for {feature_name}: ")
    input_features.append(value)

# Add a prediction button
if st.button('Predict'):
    # Convert input features to a NumPy array
    input_features_array = np.array(input_features).reshape(1, -1)

    # Make predictions using the loaded model
    predicted_class = trained_model.predict(input_features_array)

    st.write(f"Predicted class: {predicted_class[0]}")
