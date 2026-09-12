# NYC Airbnb Price Prediction

End-to-end ML project predicting Airbnb nightly prices in NYC, deployed as a Streamlit app.

## Dataset
Kaggle "New York City Airbnb Open Data" (AB_NYC_2019.csv) — ~49,000 listings, 16 original features.

## Data Preparation
- Removed 11 listings with price = $0 (invalid) and 14 with minimum_nights > 365 (data errors)
- Removed extreme price outliers (>$1000, ~top 1%) and log-transformed price to correct right-skew
- Filled reviews_per_month missing values (10,052 rows) with 0 — these correspond exactly to listings with 0 reviews, not random missingness
- Engineered: days_since_last_review, never_reviewed flag, distance_to_center (from lat/long), neighbourhood_freq (frequency encoding for 221 unique neighbourhoods)
- One-hot encoded neighbourhood_group and room_type

## Models Compared
| Model | Test R² | Test MAE ($) |
|---|---|---|
| Linear Regression | 0.56 | $48.82 |
| Random Forest (untuned) | 0.65 | — |
| Gradient Boosting | 0.64 | — |
| **Random Forest (tuned)** | **0.65** | **$43.57** |

Tuned via RandomizedSearchCV (20 candidates, 5-fold CV). Reduced train/test overfitting gap from 0.18 to 0.13 while maintaining test performance.

## Application
Streamlit app (`app.py`) takes listing details (borough, neighbourhood, room type, reviews, availability) and returns an estimated nightly price using the saved pipeline.

## Limitations
- MAE of ~$43 means predictions can be meaningfully off for atypical listings
- Neighbourhood and borough are selected independently in the app, so inconsistent combinations (e.g. wrong borough for a neighbourhood) aren't validated
- Model doesn't use listing text (name/description) or images, which likely influence real pricing
- Trained on 2019 data — doesn't reflect any pricing trends since then

## Setup
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`