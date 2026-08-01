import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set clean page config (No theme-related names)
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💳",
    layout="centered"
)

# Dark Minimalist Style (Death Note Aesthetics: Crimson accents, stark dark backdrop, clean typography)
st.markdown("""
    <style>
    /* Dark stark background */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    /* Input Box and Dropdown Customization */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #1f2833 !important;
        border: 1px solid #45a29e !important;
        color: #ffffff !important;
        border-radius: 4px;
    }

    label {
        color: #66fcf1 !important;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* Primary Crimson Button */
    .stButton>button {
        width: 100%;
        background-color: #8b0000;
        color: #ffffff;
        border: 1px solid #ff0000;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #ff0000;
        color: #000000;
        box-shadow: 0 0 10px #ff0000;
        border-color: #ff0000;
    }

    /* Container Box */
    div[data-testid="stForm"] {
        border: 1px solid #1f2833;
        padding: 25px;
        border-radius: 6px;
        background-color: #121317;
    }

    /* Success / Error Banners */
    .stSuccess {
        background-color: #002b11 !important;
        color: #25d366 !important;
        border: 1px solid #25d366 !important;
    }

    .stError {
        background-color: #3b0000 !important;
        color: #ff4d4d !important;
        border: 1px solid #ff0000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained XGBoost model
@st.cache_resource
def load_model():
    with open("loan_status_using_xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Header
st.title("Loan Approval Prediction")
st.write("Enter the required financial details below to assess loan approval eligibility.")

st.markdown("---")

with st.form("loan_features_form"):
    st.subheader("Applicant Details")
    
    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.selectbox("Number of Dependents", options=[0, 1, 2, 3, 4, 5])
        
        education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
        # Map categorical to model format (Graduate: 0, Not Graduate: 1 or vice-versa)
        education_val = 0 if education == "Graduate" else 1

        self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
        self_employed_val = 1 if self_employed == "Yes" else 0

        income_annum = st.selectbox(
            "Annual Income ($)", 
            options=[200000, 500000, 1000000, 3000000, 5000000, 7000000, 10000000]
        )

        loan_amount = st.selectbox(
            "Requested Loan Amount ($)", 
            options=[500000, 1000000, 2000000, 5000000, 10000000, 15000000, 20000000]
        )

        loan_term = st.selectbox(
            "Loan Term (Years)", 
            options=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        )

    with col2:
        cibil_score = st.selectbox(
            "CIBIL / Credit Score", 
            options=list(range(300, 901, 50))
        )

        residential_assets_value = st.selectbox(
            "Residential Assets Value ($)", 
            options=[0, 500000, 1000000, 2000000, 5000000, 10000000]
        )

        commercial_assets_value = st.selectbox(
            "Commercial Assets Value ($)", 
            options=[0, 500000, 1000000, 2000000, 5000000, 10000000]
        )

        luxury_assets_value = st.selectbox(
            "Luxury Assets Value ($)", 
            options=[0, 500000, 1000000, 2000000, 5000000, 10000000]
        )

        bank_asset_value = st.selectbox(
            "Bank Asset Value ($)", 
            options=[0, 500000, 1000000, 2000000, 5000000, 10000000]
        )

    submit_button = st.form_submit_button("Evaluate Loan Application")

if submit_button:
    # Feature order derived directly from model metadata:
    # [no_of_dependents, education, self_employed, income_annum, loan_amount, 
    #  loan_term, cibil_score, residential_assets_value, commercial_assets_value, 
    #  luxury_assets_value, bank_asset_value]
    
    input_data = np.array([[
        no_of_dependents,
        education_val,
        self_employed_val,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    try:
        prediction = model.predict(input_data)[0]
        
        st.markdown("### Decision Result")
        if prediction == 1 or prediction == " Approved" or prediction == "Approved":
            st.success("Verdict: Loan Application Approved")
        else:
            st.error("Verdict: Loan Application Rejected")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
