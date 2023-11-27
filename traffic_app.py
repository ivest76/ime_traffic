#import libraries
import streamlit as st
import pandas as pd
import sklearn
import pickle
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import xgboost

st.title("Traffic Volume Prediction: A Machine Learning App")
st.image("traffic_image.gif")
st.subheader("Utilize our advanced machine learning app to predict"
             " traffic volume.")

st.write("Use the following form to get started")

# Reading the pickle files that we created before 
# Decision Tree
dt_pickle = open('dt_traffic.pickle', 'rb') 
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()

#AdaBoost
ad_pickle = open('ad_traffic.pickle', 'rb') 
ad_model = pickle.load(ad_pickle) 
ad_pickle.close()

#XGBoost
xg_pickle = open('xg_traffic.pickle', 'rb') 
xg_model = pickle.load(xg_pickle) 
xg_pickle.close()

# Random Forest
rf_pickle = open('rf_traffic.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()


# Loading default dataset
default_df = pd.read_csv('traffic_clean.csv')
default_df["holiday"].replace(np.nan, "None", inplace=True)
default_df["holiday"] = default_df["holiday"].astype(str)


# User input form
with st.form("user_inputs"):
    holiday = st.selectbox("Choose whether today is a designated "
                           " holiday or not", options = default_df["holiday"].unique().tolist())
    temp = st.number_input("Average temp in Kelvin", min_value=0, value=288)
    rain_1h = st.number_input("mm of rain in the hour", min_value=0.0, value=0.8)
    snow_1h = st.number_input("mm of snow in the hour", min_value=0.0, value=0.1)
    clouds_all = st.number_input("Percent of cloud cover", min_value=0, value=40)
    weather_main = st.selectbox("Choose the current weather", 
                                options = default_df["weather_main"].unique().tolist())
    month = st.selectbox("Choose month", options = default_df["month"].unique().tolist())
    day = st.selectbox("Choose day", options = default_df["day"].unique().tolist())
    hour = st.number_input("Choose hour", min_value=0, value=0)
    ml_model = st.selectbox("Select model", options = ["Decision Tree", "Random Forest",
                                                    "AdaBoost", "XGBoost"],
                        placeholder = 'Choose an option')
    st.form_submit_button()

encode_df = default_df.copy()
encode_df = encode_df.drop(columns = ["traffic_volume"])
# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month,
                                 day, hour]
# Create dummies for encode_df
cat_var = ["holiday", "weather_main", "month", "day"]
encode_dummy_df = pd.get_dummies(encode_df, columns = cat_var)
# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)


if ml_model == "Decision Tree":
    # Using DT to predict() with encoded user data
    new_prediction_dt = dt_model.predict(user_encoded_df)
    # Show the predicted cost range on the app
    st.write("Decision Tree Traffic Prediction: {}".format(*new_prediction_dt))
    
elif ml_model == "AdaBoost":
    # Using AdaBoost to predict() with encoded user data
    new_prediction_dt = ad_model.predict(user_encoded_df)
    # Show the predicted cost range on the app
    st.write("AdaBoost Traffic Prediction: {}".format(*new_prediction_dt))

elif ml_model == "XGBoost":
    # Using AdaBoost to predict() with encoded user data
    new_prediction_dt = xg_model.predict(user_encoded_df)
    # Show the predicted cost range on the app
    st.write("XGBoost Traffic Prediction: {}".format(*new_prediction_dt))
       
else:
    # Using RF to predict() with encoded user data
    new_prediction_rf = rf_model.predict(user_encoded_df)
    # Show the predicted cost range on the app
    st.write("Random Forest Prediction: {}".format(*new_prediction_rf))


