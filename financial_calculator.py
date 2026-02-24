import streamlit as st
import numpy as np
import numpy_financial as npf

# Page config
st.set_page_config(page_title="Financial Calculator", page_icon="💰", layout="centered")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }
    .main {
        background-color: #0f1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
    }
    .result-box {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border: 1px solid #2d5986;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        text-align: center;
    }
    .result-label {
        color: #7eb8f7;
        font-size: 14px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-value {
        color: #ffffff;
        font-size: 42px;
        font-family: 'DM Serif Display', serif;
        font-weight: 400;
    }
    .stSelectbox label, .stNumberInput label {
        color: #a0aec0 !important;
        font-size: 13px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0, #1a4a7a);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 15px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce, #2b6cb0);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(43, 108, 176, 0.4);
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d5986, transparent);
        margin: 20px 0;
    }
    .info-text {
        color: #718096;
        font-size: 13px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#ffffff; margin-bottom:4px;'>Financial Calculator</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#7eb8f7; margin-bottom:28px;'>Time Value of Money</p>", unsafe_allow_html=True)

# Solve for selection
solve_for = st.selectbox(
    "Solve For",
    ["Present Value (PV)", "Future Value (FV)", "Interest Rate (Rate)", "Payment per Period (PMT)", "Number of Periods (NPER)"]
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Input fields based on what we're solving for
with col1:
    if solve_for != "Present Value (PV)":
        pv = st.number_input("Present Value (PV)", value=10000.0, step=100.0, format="%.2f")
    else:
        pv = None

    if solve_for != "Future Value (FV)":
        fv = st.number_input("Future Value (FV)", value=20000.0, step=100.0, format="%.2f")
    else:
        fv = None

    if solve_for != "Interest Rate (Rate)":
        rate_pct = st.number_input("Annual Interest Rate (%)", value=5.0, min_value=0.0, step=0.1, format="%.2f")
        rate = rate_pct / 100
    else:
        rate = None
        rate_pct = None

with col2:
    if solve_for != "Payment per Period (PMT)":
        pmt = st.number_input("Payment per Period (PMT)", value=0.0, step=100.0, format="%.2f",
                               help="Use negative for outgoing payments")
    else:
        pmt = None

    if solve_for != "Number of Periods (NPER)":
        nper = st.number_input("Number of Periods (NPER)", value=10.0, min_value=0.1, step=1.0, format="%.1f")
    else:
        nper = None

    period_type = st.selectbox("Payment Timing", ["End of Period (Ordinary)", "Beginning of Period (Annuity Due)"])
    when = 0 if "End" in period_type else 1

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Calculate button
if st.button("Calculate"):
    try:
        result = None
        label = ""
        prefix = "$"
        suffix = ""

        if solve_for == "Present Value (PV)":
            result = npf.pv(rate=rate, nper=nper, pmt=-pmt, fv=-fv, when=when)
            label = "Present Value"

        elif solve_for == "Future Value (FV)":
            result = npf.fv(rate=rate, nper=nper, pmt=-pmt, pv=-pv, when=when)
            label = "Future Value"

        elif solve_for == "Interest Rate (Rate)":
            result = npf.rate(nper=nper, pmt=-pmt, pv=-pv, fv=fv, when=when) * 100
            label = "Annual Interest Rate"
            prefix = ""
            suffix = "%"

        elif solve_for == "Payment per Period (PMT)":
            result = npf.pmt(rate=rate, nper=nper, pv=-pv, fv=fv, when=when)
            label = "Payment per Period"

        elif solve_for == "Number of Periods (NPER)":
            result = npf.nper(rate=rate, pmt=-pmt, pv=-pv, fv=fv, when=when)
            label = "Number of Periods"
            prefix = ""
            suffix = " periods"

        if result is not None and not np.isnan(result) and not np.isinf(result):
            formatted = f"{prefix}{result:,.2f}{suffix}" if prefix == "$" else f"{result:,.4f}{suffix}"
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-label'>{label}</div>
                <div class='result-value'>{formatted}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Could not calculate a valid result. Please check your inputs.")

    except Exception as e:
        st.error(f"Calculation error: {str(e)}. Please verify your inputs make financial sense.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>💡 For outgoing payments (loans, investments), use negative values for PMT and PV.</p>", unsafe_allow_html=True)
