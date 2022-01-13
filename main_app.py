import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

data = pd.read_csv('trained_data.csv')
data['IPS'].unique()

# Title
st.title("Laptop Price Prediction System")

# Laptop Brand
company = st.selectbox('Brand', data['Company'].unique())

# Laptop Type
type = st.selectbox('Type', data['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Laptop OS
os = st.selectbox('Operating System', data['OpSys'].unique())

# Laptop Weight
weight = st.number_input('Weight')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS Screen
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Screen Size')

# Laptop Screen Resolution
res = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# Laptop Processor
processor = st.selectbox('Processor', data['Processor'].unique())

# HDD
hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU Manufacturer', data['Gpu_manu'].unique())

if st.button('Predict Price'):

    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(res.split('x')[0])
    Y_res = int(res.split('x')[1])

    ppi = ((X_res**2)+(Y_res**2))**0.5 / (screen_size)

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, processor, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    prediction = int(np.exp(rf.predict(query)[0]))

    st.title("The Predicted price for a laptop with the selected specifications would be " + "RM " + str(prediction-200) + " to " + "RM " + str(prediction + 200))
