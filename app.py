import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBRegressor


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load model, preprocessor and training data
model = XGBRegressor()
model.load_model("xgb_model.json")
preprocessor = joblib.load("preprocessor.pkl")
df = pd.read_csv("train.csv")

# Separate input features
X = df.drop("SalePrice", axis=1)

st.title("🏠 House Price Predictor")
st.write("Enter the house details to predict its price")

# Important user inputs
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=100, value=1500)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000)
    garage_cars = st.slider("Garage Cars", 0, 4, 2)

with col2:
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=800)
    full_bath = st.slider("Full Bathrooms", 0, 5, 2)

    neighborhoods = sorted(df["Neighborhood"].dropna().unique())
    neighborhood = st.selectbox("Neighborhood", neighborhoods)

# Create one complete default house using the dataset
input_data = {}

for column in X.columns:

    if pd.api.types.is_numeric_dtype(X[column]):
        input_data[column] = X[column].median()
    else:
        input_data[column] = X[column].mode()[0]

# Replace defaults with user inputs
input_data["OverallQual"] = overall_qual
input_data["GrLivArea"] = gr_liv_area
input_data["YearBuilt"] = year_built
input_data["TotalBsmtSF"] = total_bsmt_sf
input_data["GarageCars"] = garage_cars
input_data["FullBath"] = full_bath
input_data["Neighborhood"] = neighborhood

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict House Price"):

    input_df["HasGarage"] = (input_df["GarageCars"] > 0).astype(int)

    processed_input = preprocessor.transform(input_df)

    prediction = model.predict(processed_input)[0]

    st.success(f"🏠 Estimated House Price: ${prediction:,.2f}")