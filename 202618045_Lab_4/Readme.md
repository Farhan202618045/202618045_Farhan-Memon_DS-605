# NYC Airbnb Price Prediction

End-to-end ML project predicting Airbnb nightly prices in New York City, deployed as an interactive Streamlit app.

**Live App:** https://202618045mllab04farhan.streamlit.app/

## Dataset
Kaggle "New York City Airbnb Open Data" (`AB_NYC_2019.csv`) — ~48,900 listings, 16 original features covering location, host, pricing, and review activity.

## Data Preparation
- Removed 11 listings with `price = $0` (invalid — not a real nightly rate) and 14 listings with `minimum_nights > 365` (data errors)
- Removed extreme price outliers (>$1,000, ~top 1% of listings) and log-transformed price to correct heavy right-skew, so the model trains on a roughly bell-shaped target
- Filled 10,052 missing `reviews_per_month` values with 0 — confirmed these correspond exactly to listings with `number_of_reviews == 0`, so 0 is the correct value, not an estimate
- Filled small numbers of missing `name`/`host_name` values with `'Unknown'`
- Engineered features: `days_since_last_review` (with a sentinel value for never-reviewed listings), `never_reviewed` flag, `distance_to_center` (from latitude/longitude), `neighbourhood_freq` (frequency encoding for 221 unique neighbourhoods)
- One-hot encoded `neighbourhood_group` (borough) and `room_type`, with `drop_first=True` to avoid multicollinearity

## Models Compared

| Model | Test R² | Test RMSE ($) | Test MAE ($) |
|---|---|---|---|
| Linear Regression | 0.56 | $90.82 | $48.82 |
| Random Forest (untuned) | 0.65 | — | — |
| Gradient Boosting (default) | 0.64 | — | — |
| **Random Forest (tuned, final)** | **0.65** | **$82.69** | **$43.57** |

The final model was tuned via `RandomizedSearchCV` (20 candidates, 5-fold CV), which reduced the train/test overfitting gap from 0.18 to 0.13 while slightly *improving* test R² — a genuine generalization win, not a trade-off. It was then retrained with fewer trees (100 vs. 300) to shrink the saved pipeline from ~100MB to ~16MB for deployment, at a negligible cost to accuracy (R² 0.6517 → 0.6466).

## Application
The Streamlit app (`app.py`) takes listing details — borough, neighbourhood, room type, minimum nights, review activity, host's listing count, and availability — and returns an estimated nightly price using the saved end-to-end pipeline (scaler + model bundled together via `joblib`).

## Limitations
- MAE of ~$44 means predictions can be meaningfully off for atypical or luxury listings
- Borough and neighbourhood are selected independently in the app and aren't cross-validated, so a user can pick an inconsistent combination (e.g., a Queens neighbourhood under the Manhattan borough filter) without an error
- The model doesn't use listing text (name/description) or photos, which likely influence real-world pricing
- Trained on 2019 data — doesn't reflect any pricing shifts since then (e.g., post-pandemic short-term rental trends)
- New/unreviewed listings default to a fixed "days since last review" sentinel in the app rather than asking the user directly, as a UX simplification

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `202618045_Lab04.ipynb` — full analysis, feature engineering, and model training notebook
- `app.py` — Streamlit application
- `airbnb_price_pipeline.pkl` — trained preprocessing + model pipeline
- `neighbourhood_freq_map.pkl`, `neighbourhood_coords_map.pkl` — lookup tables used by the app
- `requirements.txt` — dependencies