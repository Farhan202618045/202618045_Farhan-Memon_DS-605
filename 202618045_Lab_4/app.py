import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline = joblib.load(os.path.join(BASE_DIR, 'airbnb_price_pipeline.pkl'))
neighbourhood_freq_map = joblib.load(os.path.join(BASE_DIR, 'neighbourhood_freq_map.pkl'))
neighbourhood_coords_map = joblib.load(os.path.join(BASE_DIR, 'neighbourhood_coords_map.pkl'))

# This must match X_train's column order EXACTLY (from your notebook's X.columns.tolist())
FEATURE_ORDER = [
    'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
    'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
    'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island',
    'room_type_Private room', 'room_type_Shared room',
    'neighbourhood_freq', 'days_since_last_review', 'never_reviewed',
    'distance_to_center'
]

# Same sentinel value used during training for "never reviewed" (max days_since_last_review + 1)
SENTINEL_DAYS = 3026
CENTER_LAT, CENTER_LON = 40.7580, -73.9855

st.title("🏠 NYC Airbnb Price Predictor")
st.write("Enter listing details to estimate a fair nightly price.")

neighbourhood_group = st.selectbox("Borough", ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
neighbourhood = st.selectbox("Neighbourhood", sorted(neighbourhood_coords_map.index.tolist()))
room_type = st.selectbox("Room Type", ['Entire home/apt', 'Private room', 'Shared room'])

minimum_nights = st.number_input("Minimum Nights", min_value=1, value=3)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=0)
reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=0.0, step=0.1)
calculated_host_listings_count = st.number_input("Host's Total Listings", min_value=1, value=1)
availability_365 = st.slider("Availability (days/year)", 0, 365, 180)

if st.button("Predict Price"):
    # Look up coordinates and frequency for the chosen neighbourhood
    lat = neighbourhood_coords_map.loc[neighbourhood, 'latitude']
    lon = neighbourhood_coords_map.loc[neighbourhood, 'longitude']
    freq = neighbourhood_freq_map.get(neighbourhood, 1)  # fallback to 1 if unseen
    distance = np.sqrt((lat - CENTER_LAT)**2 + (lon - CENTER_LON)**2)

    never_reviewed = 1 if number_of_reviews == 0 else 0
    days_since_last_review = SENTINEL_DAYS if never_reviewed else 30  # simple default for a reviewed listing

    # Build the one-hot columns manually, matching training's drop_first=True behavior
    row = {
        'latitude': lat,
        'longitude': lon,
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'calculated_host_listings_count': calculated_host_listings_count,
        'availability_365': availability_365,
        'neighbourhood_group_Brooklyn': 1 if neighbourhood_group == 'Brooklyn' else 0,
        'neighbourhood_group_Manhattan': 1 if neighbourhood_group == 'Manhattan' else 0,
        'neighbourhood_group_Queens': 1 if neighbourhood_group == 'Queens' else 0,
        'neighbourhood_group_Staten Island': 1 if neighbourhood_group == 'Staten Island' else 0,
        'room_type_Private room': 1 if room_type == 'Private room' else 0,
        'room_type_Shared room': 1 if room_type == 'Shared room' else 0,
        'neighbourhood_freq': freq,
        'days_since_last_review': days_since_last_review,
        'never_reviewed': never_reviewed,
        'distance_to_center': distance
    }

    input_df = pd.DataFrame([row])[FEATURE_ORDER]  # enforce exact column order

    pred_log = pipeline.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"### Estimated Price: ${pred_price:.2f} / night")