import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
}
.main-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 50%, #ff6fd8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #a0a0c0;
    font-size: 1.05rem;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}
.glass-card h3 {
    margin-top: 0;
    font-weight: 700;
    font-size: 1.15rem;
    color: #e0e0ff;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}

/* Result cards */
.result-card {
    text-align: center;
    border-radius: 20px;
    padding: 2.5rem 1.5rem;
    margin-top: 1rem;
}
.result-cancel {
    background: linear-gradient(135deg, rgba(255,60,80,0.18) 0%, rgba(255,60,80,0.06) 100%);
    border: 1.5px solid rgba(255,60,80,0.35);
}
.result-notcancel {
    background: linear-gradient(135deg, rgba(0,210,130,0.18) 0%, rgba(0,210,130,0.06) 100%);
    border: 1.5px solid rgba(0,210,130,0.35);
}
.result-card .emoji { font-size: 3.5rem; }
.result-card .label {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-cancel .label { color: #ff3c50; }
.result-notcancel .label { color: #00d282; }
.result-card .prob {
    font-size: 1rem;
    color: #a0a0c0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13112b 0%, #1d1b3a 100%);
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(123,47,247,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_hotel_demand_030526_11:09.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────────────────────────
# Categorical options
# ──────────────────────────────────────────────────────────────────
HOTEL_OPTIONS = ['Resort Hotel', 'City Hotel']
MONTH_OPTIONS = ['January','February','March','April','May','June',
                 'July','August','September','October','November','December']
MEAL_OPTIONS = ['BB', 'FB', 'HB', 'SC', 'Undefined']
MARKET_SEGMENT_OPTIONS = ['Direct','Corporate','Online TA','Offline TA/TO',
                          'Groups','Complementary','Aviation']
DISTRIBUTION_CHANNEL_OPTIONS = ['Direct','Corporate','TA/TO','Undefined','GDS']
ROOM_TYPE_OPTIONS = ['A','B','C','D','E','F','G','H','L']
DEPOSIT_TYPE_OPTIONS = ['No Deposit','Refundable','Non Refund']
CUSTOMER_TYPE_OPTIONS = ['Transient','Contract','Transient-Party','Group']

COUNTRY_OPTIONS = [
    'GBR','PRT','USA','ESP','IRL','FRA','ROU','NOR','OMN','ARG','POL','DEU',
    'BEL','CHE','CN','GRC','ITA','NLD','DNK','RUS','SWE','AUS','EST','CZE',
    'BRA','FIN','MOZ','BWA','LUX','SVN','ALB','IND','CHN','MEX','MAR','UKR',
    'SMR','LVA','PRI','SRB','CHL','AUT','BLR','LTU','TUR','ZAF','ISR','CYM',
    'ZMB','CPV','ZWE','DZA','KOR','CRI','HUN','ARE','TUN','JAM','HRV','HKG',
    'IRN','GEO','AND','GIB','URY','JEY','CAF','CYP','COL','GGY','KWT','NGA',
    'MDV','VEN','SVK','AGO','FJI','KAZ','PAK','IDN','LBN','PHL','SEN','SYC',
    'AZE','BHR','NZL','THA','DOM','MKD','MYS','ARM','JPN','LKA','CUB','CMR',
    'BIH','MUS','COM','SUR','UGA','BGR','CIV','JOR','SYR','SGP','BDI','SAU',
    'VNM','PLW','EGY','PER','MLT','MWI','ECU','MDG','ISL','UZB','NPL','BHS',
    'MAC','TGO','TWN','DJI','STP','KNA','ETH','IRQ','HND','RWA','QAT','KHM',
    'MCO','BGD','IMN','TJK','NIC','BEN','VGB','TZA','GAB','GHA','TMP','GLP',
    'KEN','LIE','GNB','MNE','UMI','MYT','FRO','MMR','PAN','BFA','LBY','MLI',
    'NAM','BOL','PRY','BRB','ABW','AIA','SLV','DMA','PYF','GUY','LCA','ATA',
    'GTM','ASM','MRT','NCL','KIR','SDN','ATF','SLE','LAO',
]

# ──────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏨 Hotel Booking Cancellation Predictor</h1>
    <p>Predict whether a hotel booking will be cancelled using an XGBoost pipeline model</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Input form — grouped into logical sections
# ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    # ── 1. Hotel & Arrival Info ──
    st.markdown('<div class="glass-card"><h3>🏢 Hotel & Arrival Info</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    hotel = c1.selectbox("Hotel Type", HOTEL_OPTIONS)
    arrival_date_year = c2.number_input("Arrival Year", min_value=2015, max_value=2030, value=2025, step=1)
    c3, c4 = st.columns(2)
    arrival_date_month = c3.selectbox("Arrival Month", MONTH_OPTIONS)
    lead_time = c4.number_input("Lead Time (days)", min_value=0, max_value=800, value=30, step=1)

    # ── 2. Stay Duration ──
    st.markdown('<div class="glass-card"><h3>🌙 Stay Duration</h3></div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    stays_in_weekend_nights = c5.number_input("Weekend Nights", min_value=0, max_value=20, value=1, step=1)
    stays_in_week_nights = c6.number_input("Week Nights", min_value=0, max_value=50, value=2, step=1)

    # ── 3. Guest Info ──
    st.markdown('<div class="glass-card"><h3>👥 Guest Info</h3></div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    adults = c7.number_input("Adults", min_value=0, max_value=10, value=2, step=1)
    children = c8.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    babies = c9.number_input("Babies", min_value=0, max_value=10, value=0, step=1)

    # ── 4. Meal & Country ──
    st.markdown('<div class="glass-card"><h3>🍽️ Meal & Country</h3></div>', unsafe_allow_html=True)
    c10, c11 = st.columns(2)
    meal = c10.selectbox("Meal Plan", MEAL_OPTIONS)
    country = c11.selectbox("Country (ISO 3166)", COUNTRY_OPTIONS)

with col_right:
    # ── 5. Booking Channel ──
    st.markdown('<div class="glass-card"><h3>📡 Booking Channel</h3></div>', unsafe_allow_html=True)
    c12, c13 = st.columns(2)
    market_segment = c12.selectbox("Market Segment", MARKET_SEGMENT_OPTIONS)
    distribution_channel = c13.selectbox("Distribution Channel", DISTRIBUTION_CHANNEL_OPTIONS)

    # ── 6. Room & Deposit ──
    st.markdown('<div class="glass-card"><h3>🛏️ Room & Deposit</h3></div>', unsafe_allow_html=True)
    c14, c15 = st.columns(2)
    reserved_room_type = c14.selectbox("Reserved Room Type", ROOM_TYPE_OPTIONS)
    deposit_type = c15.selectbox("Deposit Type", DEPOSIT_TYPE_OPTIONS)

    # ── 7. Customer & Booking History ──
    st.markdown('<div class="glass-card"><h3>📋 Customer & Booking History</h3></div>', unsafe_allow_html=True)
    customer_type = st.selectbox("Customer Type", CUSTOMER_TYPE_OPTIONS)
    c16, c17 = st.columns(2)
    previous_cancellations = c16.number_input("Previous Cancellations", min_value=0, max_value=30, value=0, step=1)
    previous_bookings_not_canceled = c17.number_input("Previous Bookings (Not Cancelled)", min_value=0, max_value=80, value=0, step=1)
    c18, c19 = st.columns(2)
    booking_changes = c18.number_input("Booking Changes", min_value=0, max_value=30, value=0, step=1)
    days_in_waiting_list = c19.number_input("Days in Waiting List", min_value=0, max_value=500, value=0, step=1)

    # ── 8. Financial & Extras ──
    st.markdown('<div class="glass-card"><h3>💰 Financial & Extras</h3></div>', unsafe_allow_html=True)
    c20, c21 = st.columns(2)
    adr = c20.number_input("ADR (Avg Daily Rate)", min_value=0.0, max_value=5000.0, value=100.0, step=0.5, format="%.2f")
    required_car_parking_spaces = c21.number_input("Car Parking Spaces", min_value=0, max_value=10, value=0, step=1)
    c22, c23, c24 = st.columns(3)
    total_of_special_requests = c22.number_input("Special Requests", min_value=0, max_value=10, value=0, step=1)
    is_by_agent = c23.selectbox("Booked via Agent?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_by_company = c24.selectbox("Booked via Company?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# ──────────────────────────────────────────────────────────────────
# Predict button
# ──────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔮 Predict Cancellation"):
    # Build input dataframe (column order must match training)
    input_dict = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "is_by_agent": is_by_agent,
        "is_by_company": is_by_company,
    }

    input_df = pd.DataFrame([input_dict])

    try:
        model = load_model()
        prediction = model.predict(input_df)[0]

        # Try to get probability (the inner estimator pipeline)
        try:
            proba = model.estimator_.predict_proba(input_df)[0]
            prob_cancel = proba[1]
            prob_not = proba[0]
        except Exception:
            prob_cancel = None
            prob_not = None

        res_left, res_mid, res_right = st.columns([1, 2, 1])
        with res_mid:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card result-cancel">
                    <div class="emoji">❌</div>
                    <div class="label">CANCELLED</div>
                    <div class="prob">{"Probability: {:.1%}".format(prob_cancel) if prob_cancel is not None else ""}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-notcancel">
                    <div class="emoji">✅</div>
                    <div class="label">NOT CANCELLED</div>
                    <div class="prob">{"Probability: {:.1%}".format(prob_not) if prob_not is not None else ""}</div>
                </div>
                """, unsafe_allow_html=True)

        # Show input summary in expander
        with st.expander("📄 Input Data Summary"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    except FileNotFoundError:
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`. Please place the .pkl file in the same directory as this app.")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ──────────────────────────────────────────────────────────────────
# Sidebar — model info
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ Model Info")
    st.markdown("""
    | Property | Value |
    |---|---|
    | **Model** | XGBClassifier |
    | **Wrapper** | TunedThresholdClassifierCV |
    | **n_estimators** | 300 |
    | **max_depth** | 6 |
    | **learning_rate** | 0.1 |
    | **Scoring** | F-beta (β=0.5) |
    """)

    st.markdown("### 🔧 Preprocessing")
    st.markdown("""
    - **RobustScaler** → numerical columns
    - **OneHotEncoder** (drop first) → `hotel`, `deposit_type`, `customer_type`
    - **BinaryEncoder** → `meal`, `distribution_channel`, `market_segment`, `reserved_room_type`, `arrival_date_month`, `country`
    """)

    st.markdown("---")
    st.caption("Hotel Booking Cancellation Predictor v1.0")
