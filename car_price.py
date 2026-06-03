
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# ========== CUSTOM CSS ==========
st.markdown("""
<style>

/* Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Glass Effect Container */
.main {
    background-color: rgba(0,0,0,0.70);
    padding: 25px;
    border-radius: 15px;
}

/* TEXT VISIBILITY FIX */
h1, h2, h3, h4, p, label, div {
    color: #f5f5f5 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.85);
}

/* BIG TITLE */
h1 {
    font-size: 44px !important;
    text-align: center;
    font-weight: bold;
}

/* BUTTON STYLE */
.stButton>button {
    background-color: #00c6ff;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 18px;
}

.stButton>button:hover {
    background-color: #0072ff;
    color: white;
}

/* INPUT FIELDS */
.stNumberInput input, .stSelectbox div {
    font-size: 16px !important;
    color: #000000 !important;
    background-color: #ffffff !important;
    border-radius: 6px !important;
}

</style>
""", unsafe_allow_html=True)

# ========== TITLE ==========
st.title("🚗 Car Price Prediction Dashboard")
st.markdown("### Predict used car price using Machine Learning")

# ========== LOAD DATA ==========
df = pd.read_csv("car_price.csv")

df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

df["horsepower"] = df["horsepower"].astype(float)
df["peak-rpm"] = df["peak-rpm"].astype(float)
df["price"] = df["price"].astype(float)

# ========== ENCODING ==========
df["fuel-type"] = LabelEncoder().fit_transform(df["fuel-type"])
df["engine-location"] = LabelEncoder().fit_transform(df["engine-location"])
df["engine-type"] = LabelEncoder().fit_transform(df["engine-type"])

features = [
    "fuel-type",
    "engine-location",
    "engine-type",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg"
]

X = df[features]
y = df["price"]

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

# ========== SIDEBAR INPUT ==========
st.sidebar.header("🧾 Car Details")

fuel_type = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])
engine_location = st.sidebar.selectbox("Engine Location", ["front", "rear"])
engine_type = st.sidebar.selectbox("Engine Type", ["dohc", "ohc", "ohcv", "l", "rotor"])

horsepower = st.sidebar.number_input("Horsepower", 40, 300, 100)
peak_rpm = st.sidebar.number_input("Peak RPM", 4000, 7000, 5000)
city_mpg = st.sidebar.number_input("City MPG", 10, 60, 25)
highway_mpg = st.sidebar.number_input("Highway MPG", 10, 60, 30)

# ========== MAPPING ==========
fuel_map = {"diesel": 0, "gas": 1}
engine_loc_map = {"front": 0, "rear": 1}
engine_type_map = {"dohc": 0, "l": 1, "ohc": 2, "ohcv": 3, "rotor": 4}

input_data = pd.DataFrame([[
    fuel_map[fuel_type],
    engine_loc_map[engine_location],
    engine_type_map[engine_type],
    horsepower,
    peak_rpm,
    city_mpg,
    highway_mpg
]], columns=features)

# ========== PREDICTION ==========
st.markdown("## 💰 Prediction Result")

if st.button("Predict Price 🚗"):
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f}")
    st.metric("Predicted Price", f"₹ {prediction[0]:,.2f}")

    # Graph
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(["Price"], [prediction[0]])
    ax.set_ylabel("₹ Price")
    st.pyplot(fig)

# ========== FEATURE IMPORTANCE ==========
st.markdown("## 📊 Feature Importance")

fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.barh(features, model.feature_importances_)
ax2.set_xlabel("Importance")
st.pyplot(fig2)

# ========== DOWNLOAD DATASET ==========
st.markdown("## 📥 Download Dataset")

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇ Download Clean Dataset",
    data=csv,
    file_name="clean_car_price.csv",
    mime="text/csv"
)

# ========== PREVIEW ==========
st.markdown("## 📌 Dataset Preview")
st.dataframe(df.head(10))