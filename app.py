import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open('rf1.pkl', 'rb'))

# Function to scale the input features
def scale_features(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

# Define the Streamlit app
def main():
    st.title("Car Price Prediction")

    st.write("### Enter the details of the car")

    year = st.number_input('Year', min_value=1990, max_value=2023, value=2015)
    km_driven = st.number_input('Kilometers Driven', min_value=0, value=50000)
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    owner = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

    transmission = 1 if transmission == 'Manual' else 0

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'transmission': [transmission],
        'fuel_Diesel': [1 if fuel == 'Diesel' else 0],
        'fuel_Electric': [1 if fuel == 'Electric' else 0],
        'fuel_LPG': [1 if fuel == 'LPG' else 0],
        'fuel_Petrol': [1 if fuel == 'Petrol' else 0],
        'seller_type_Individual': [1 if seller_type == 'Individual' else 0],
        'seller_type_Trustmark Dealer': [1 if seller_type == 'Trustmark Dealer' else 0],
        'owner_Fourth & Above Owner': [1 if owner == 'Fourth & Above Owner' else 0],
        'owner_Second Owner': [1 if owner == 'Second Owner' else 0],
        'owner_Test Drive Car': [1 if owner == 'Test Drive Car' else 0],
        'owner_Third Owner': [1 if owner == 'Third Owner' else 0]
    })

    # Scale the input data
    input_data_scaled = scale_features(input_data)

    if st.button('Predict'):
        prediction = model.predict(input_data_scaled)
        st.write(f"### Predicted Selling Price: {prediction[0]:.2f}")

if __name__ == '__main__':
    main()
