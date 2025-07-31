import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration early
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection System")

# Use st.cache_resource for loading the model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Use st.cache_data for loading and preprocessing reference data
@st.cache_data
def load_and_preprocess_data(data_path):
    try:
        df_ref = pd.read_csv(data_path)

        if 'Unnamed: 0' in df_ref.columns:
            df_ref = df_ref.drop(columns=['Unnamed: 0'], errors='ignore')
        if 'is_fraud' in df_ref.columns:
            df_ref = df_ref.drop(columns=['is_fraud'], errors='ignore')

        df_encoded = pd.get_dummies(df_ref, drop_first=True)
        expected_cols = df_encoded.columns.tolist()
        if 'Unnamed: 0' not in expected_cols:

            expected_cols.insert(0, 'Unnamed: 0')


        amount_threshold = df_ref['amount'].quantile(0.99)
        amount_mean = df_ref['amount'].mean()
        amount_std = df_ref['amount'].std()
        distance_threshold = df_ref['distance_from_home'].quantile(0.95)

        return df_ref, expected_cols, amount_threshold, amount_mean, amount_std, distance_threshold

    except FileNotFoundError:
        st.error(f"Reference data file not found at {data_path}. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or processing reference data: {e}")
        st.stop()

# Define paths
model_path = 'XGBoost_model.joblib'
data_path = 'df_final_2.csv'

# Load model and data using cached functions
model = load_model(model_path)
df_ref, expected_cols, amount_threshold, amount_mean, amount_std, distance_threshold = load_and_preprocess_data(data_path)

# UI setup
st.sidebar.header("Enter Transaction Details")

def get_options(col):
    if col in df_ref.columns and df_ref[col].dtype == 'object':
        return sorted(df_ref[col].dropna().unique().tolist())
    return []

merchant_category = st.sidebar.selectbox("Merchant Category", get_options('merchant_category'))
merchant_type = st.sidebar.selectbox("Merchant Type", get_options('merchant_type'))
currency = st.sidebar.selectbox("Currency", get_options('currency'))
country = st.sidebar.selectbox("Country", get_options('country'))
card_type = st.sidebar.selectbox("Card Type", get_options('card_type'))
device = st.sidebar.selectbox("Device", get_options('device'))
channel = st.sidebar.selectbox("Channel", get_options('channel'))

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=50.0)
card_present = st.sidebar.selectbox("Card Present", [1, 0])
distance = st.sidebar.number_input("Distance from Home", min_value=0, value=10)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

# Feature engineering
is_night = int(hour < 6 or hour >= 22)
is_peak = int(12 <= hour <= 20)
hour_bin = (
    'morning' if 6 <= hour < 12 else
    'afternoon' if 12 <= hour < 17 else
    'evening' if 17 <= hour < 22 else
    'night'
)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

is_large = int(amount > amount_threshold)
log_amount = np.log1p(amount)
zscore = (amount - amount_mean) / amount_std
is_remote = int(distance > distance_threshold)
card_not_present = 1 - card_present
risk_score = 0  # Placeholder - If this is a dynamic calculation, it should be based on input, not a static 0
combo = f"{channel}_{device}"

# Create input DataFrame
input_data = pd.DataFrame([{
    'merchant_category': merchant_category,
    'merchant_type': merchant_type,
    'amount': amount,
    'currency': currency,
    'country': country,
    'card_type': card_type,
    'card_present': card_present,
    'device': device,
    'channel': channel,
    'distance_from_home': distance,
    'hour': hour,
    'is_night': is_night,
    'is_peak_hour': is_peak,
    'hour_bin': hour_bin,
    'hour_sin': hour_sin,
    'hour_cos': hour_cos,
    'is_large_amount': is_large,
    'log_amount': log_amount,
    'amount_zscore': zscore,
    'is_remote': is_remote,
    'is_card_not_present': card_not_present,
    'device_risk_score': risk_score, # Consider how 'risk_score' is truly derived. If it's always 0, it's not adding value.
    'channel_device_combo': combo
}])

encoded_input = pd.get_dummies(input_data, drop_first=True)

# Ensure required one-hot columns are present in the input
# This part is crucial for aligning with the model's expected features.
# Make sure to handle all possible categorical values that the model was trained on.
# Iterating through expected_cols and adding missing ones is a robust approach.
for col in expected_cols:
    if col not in encoded_input.columns:
        encoded_input[col] = 0

# Ensure the order of columns matches the training data
processed_input = encoded_input[expected_cols]

# Prediction
if st.sidebar.button("Predict Fraud"):
    try:
        pred = model.predict(processed_input)
        prob = model.predict_proba(processed_input)[:, 1][0]

        st.subheader("Prediction Result")
        if pred[0] == 1:
            st.error("🔴 Fraudulent Transaction Detected!")
        else:
            st.success("🟢 Legitimate Transaction")

        st.write(f"Fraud Probability: **{prob:.4f}**")
        st.markdown("----")
        st.subheader("Transaction Summary")
        st.write(input_data)

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")