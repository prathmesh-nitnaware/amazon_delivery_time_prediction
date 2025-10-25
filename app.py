import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- 1. Load the Saved Model and Columns ---
try:
    # Load the trained LightGBM model
    model = joblib.load('best_delivery_model.joblib')
    # Load the list of feature columns the model was trained on
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'best_delivery_model.joblib' and 'model_columns.joblib' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

# --- 2. Create the Prediction Function ---
def predict_delivery_time(data):
    """
    Preprocesses raw input data and returns a delivery time prediction.
    Args:
        data (pd.DataFrame): A DataFrame with raw, unseen data.
    Returns:
        float: The predicted delivery time in minutes.
    """
    df = data.copy()

    # --- Replicate the exact same preprocessing pipeline ---
    # A. Robustly handle missing values (though UI prevents most)
    # The .astype(str) conversion is removed as UI provides clean time objects
    missing_values_to_replace = ['NaN', 'nan', 'None', '']
    df.replace(missing_values_to_replace, np.nan, inplace=True)

    # B. Feature Engineering
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Order_Timestamp'] = df.apply(lambda row: datetime.combine(row['Order_Date'], row['Order_Time']), axis=1)
    df['Pickup_Timestamp'] = df.apply(lambda row: datetime.combine(row['Order_Date'], row['Pickup_Time']), axis=1)

    df['Preparation_Time_mins'] = (df['Pickup_Timestamp'] - df['Order_Timestamp']).dt.total_seconds() / 60
    df['Order_Hour'] = df['Order_Timestamp'].dt.hour
    
    # C. One-hot encode categorical features
    # Convert categorical columns to 'category' dtype to be safe
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    df = pd.get_dummies(df, drop_first=True)
    
    # D. Align columns with the training data
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # --- Make the prediction ---
    prediction = model.predict(df)
    
    return prediction[0]

# --- 3. Streamlit App UI ---
st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: white;
        background-color: #ff9900;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #e68a00;
    }
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("📦 Amazon Delivery Time Predictor")
st.markdown("Enter the details of the order to get an estimated delivery time.")

# 

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    st.header("📍 Location & Distance")
    distance = st.number_input("Distance from Restaurant to Delivery (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    area = st.selectbox("Area Type", ['Urban', 'Metropolitan', 'Semi-Urban'])

    st.header("📅 Order & Pickup Timings")
    order_date = st.date_input("Order Date", datetime.now())
    order_time = st.time_input("Order Time", datetime.now().time())
    pickup_time = st.time_input("Pickup Time", (datetime.now() + pd.Timedelta(minutes=15)).time())

with col2:
    st.header("🚚 Delivery Conditions")
    weather = st.selectbox("Weather Conditions", ['Sunny', 'Cloudy', 'Rainy', 'Windy', 'Fog', 'Stormy'])
    traffic = st.selectbox("Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
    vehicle = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'electric_scooter'])

    st.header("👤 Agent Details")
    agent_age = st.slider("Delivery Agent's Age", 20, 50, 30)
    agent_rating = st.slider("Delivery Agent's Rating", 4.0, 5.0, 4.5, 0.1)

    # Hidden fields for prediction that are less common to change
    # In a real app, these might be looked up from a database
    store_lat, store_lon = 22.5726, 88.3639
    drop_lat, drop_lon = 22.5448, 88.3426
    category = "Snack"


# --- Prediction Logic ---
if st.button("Predict Delivery Time"):
    # Create a dictionary of the input data
    input_data = {
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Store_Latitude': store_lat,
        'Store_Longitude': store_lon,
        'Drop_Latitude': drop_lat,
        'Drop_Longitude': drop_lon,
        'Order_Date': order_date,
        'Order_Time': order_time,
        'Pickup_Time': pickup_time,
        'Weather': weather,
        'Traffic': traffic,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category,
        'Distance_km': distance
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Check for valid prep time
    prep_time = (datetime.combine(order_date, pickup_time) - datetime.combine(order_date, order_time)).total_seconds() / 60
    if prep_time < 0:
        st.error("Error: Pickup time cannot be before the order time.")
    else:
        try:
            with st.spinner('Calculating...'):
                predicted_time = predict_delivery_time(input_df)
            
            st.metric(
                label="Predicted Delivery Time",
                value=f"{predicted_time:.0f} minutes"
            )
            st.success("Prediction successful!")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

