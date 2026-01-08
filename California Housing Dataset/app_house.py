import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing

# 1. Setup Page Configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Predictor (XGBoost)")
st.write("Enter house details to predict the price in $100,000s")


# 2. Load and train the model (Cached for performance)
@st.cache_resource
def train_optimized_model():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Feature Engineering inside the app
    X['Rooms_per_HH'] = X['AveRooms'] / X['AveOccup']

    # Using XGBoost for better accuracy
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X, y)

    # Store means for default input values
    means = X.mean().to_dict()
    return model, data.feature_names, means


# Load the model and data info
model, feature_names, feature_means = train_optimized_model()

# 3. Create Sidebar Input Fields
st.sidebar.header("House Specifications")
user_inputs = {}

for col in feature_names:
    # We use the pre-calculated means from feature_means
    user_inputs[col] = st.sidebar.number_input(f"Enter {col}", value=float(feature_means[col]))

# Add the engineered feature manually
user_inputs['Rooms_per_HH'] = user_inputs['AveRooms'] / user_inputs['AveOccup']

# 4. Prediction Logic
if st.button("Predict Price"):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Ensure columns order matches the model
    prediction = model.predict(input_df)

    st.divider()
    st.header(f"💰 Estimated Value: ${prediction[0] * 100000:,.2f}")
    st.info("Note: Prices are in hundreds of thousands of dollars.")