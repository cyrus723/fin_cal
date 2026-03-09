import math

import streamlit as st

# Page config
st.set_page_config(
    page_title="Financial Calculator",
    page_icon="USD",
    layout="centered",
)

# Custom CSS
st.markdown(
    """
<style>
    html, body, [class*="css"] {
        font-family: sans-serif;
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
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-value {
        color: #ffffff;
        font-size: 36px;
        font-weight: 600;
    }
    .stSelectbox label, .stNumberInput label, .stRadio label {
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
        color: #a0aec0;
        font-size: 13px;
    }
</style>
""",
    unsafe_allow_html=True,
)

SOLVE_OPTIONS = {
    "Present Value (PV)": {"label": "Present Value", "prefix": "$", "suffix": ""},
    "Future Value (FV)": {"label": "Future Value", "prefix": "$", "suffix": ""},
    "Interest Rate (Rate)": {"label": "Interest Rate per Period", "prefix": "", "suffix": "%"},
    "Payment per Period (PMT)": {"label": "Payment per Period", "prefix": "$", "suffix": ""},
    "Number of Periods (NPER)": {"label": "Number of Periods", "prefix": "", "suffix": " periods"},
}

FREQUENCY_OPTIONS = {
    "Annual": 1,
    "Semiannual": 2,
    "Quarterly": 4,
    "Monthly": 12,
    "Weekly": 52,
    "Daily": 365,
}


def safe_is_valid(value):
    return value is not None and math.isfinite(value)


def pv(rate, nper, pmt, fv=0.0, when=0):
    if rate == 0:
        return -(fv + pmt * nper)
    return -(fv + pmt * (1 + rate * when) * ((1 + rate) ** nper - 1) / rate) / ((1 + rate) ** nper)


def fv(rate, nper, pmt, pv_value=0.0, when=0):
    if rate == 0:
        return -(pv_value + pmt * nper)
    return -(pv_value * (1 + rate) ** nper + pmt * (1 + rate * when) * ((1 + rate) ** nper - 1) / rate)


def pmt(rate, nper, pv_value, fv_value=0.0, when=0):
    if nper == 0:
        raise ValueError("NPER must be greater than 0.")
    if rate == 0:
        return -(fv_value + pv_value) / nper
    return -(fv_value + pv_value * (1 + rate) ** nper) * rate / (
        (1 + rate * when) * ((1 + rate) ** nper - 1)
    )


def nper(rate, pmt_value, pv_value, fv_value=0.0, when=0):
    if rate == 0:
        if pmt_value == 0:
            raise ValueError("PMT cannot be zero when rate is zero.")
        return -(fv_value + pv_value) / pmt_value

    adjusted_pmt = pmt_value * (1 + rate * when) / rate
    numerator = adjusted_pmt - fv_value
    denominator = adjusted_pmt + pv_value

    if numerator <= 0 or denominator <= 0:
        raise ValueError("Inputs do not produce a real solution for NPER.")

    return math.log(numerator / denominator) / math.log(1 + rate)


def rate_bisection(nper_value, pmt_value, pv_value, fv_value=0.0, when=0, tol=1e-10, max_iter=200):
    def balance(period_rate):
        if abs(period_rate) < 1e-14:
            return pv_value + pmt_value * nper_value + fv_value
        return (
            pv_value * (1 + period_rate) ** nper_value
            + pmt_value * (1 + period_rate * when) * ((1 + period_rate) ** nper_value - 1) / period_rate
            + fv_value
        )

    low, high = -0.9999, 10.0
    f_low, f_high = balance(low), balance(high)

    expand_count = 0
    while f_low * f_high > 0 and expand_count < 20:
        high *= 2
        f_high = balance(high)
        expand_count += 1

    if f_low * f_high > 0:
        raise ValueError("Could not find a valid rate for these inputs.")

    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = balance(mid)

        if abs(f_mid) < tol:
            return mid

        if f_low * f_mid < 0:
            high = mid
        else:
            low = mid
            f_low = f_mid

    return (low + high) / 2


def format_result(value, prefix="", suffix=""):
    if prefix == "$":
        return f"{prefix}{value:,.2f}{suffix}"
    if suffix in {"%", " periods"}:
        return f"{value:,.4f}{suffix}"
    return f"{prefix}{value:,.2f}{suffix}"


st.markdown("<h1 style='color:#ffffff; margin-bottom:4px;'>Financial Calculator</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#7eb8f7; margin-bottom:28px;'>Time Value of Money</p>", unsafe_allow_html=True)

solve_for = st.selectbox("Solve For", list(SOLVE_OPTIONS.keys()))

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pv_input = None
    fv_input = None
    rate = None

    if solve_for != "Present Value (PV)":
        pv_input = st.number_input("Present Value (PV)", value=10000.0, step=100.0, format="%.2f")

    if solve_for != "Future Value (FV)":
        fv_input = st.number_input("Future Value (FV)", value=20000.0, step=100.0, format="%.2f")

    period_frequency_label = st.selectbox(
        "Compounding / Payment Frequency",
        list(FREQUENCY_OPTIONS.keys()),
        index=3,
    )
    periods_per_year = FREQUENCY_OPTIONS[period_frequency_label]

    if solve_for != "Interest Rate (Rate)":
        rate_mode = st.radio(
            "Rate Input Type",
            ["Rate per Period", "Nominal Annual Rate"],
            horizontal=True,
        )
        if rate_mode == "Rate per Period":
            rate_pct = st.number_input(
                "Interest Rate per Period (%)",
                value=5.0,
                step=0.1,
                format="%.4f",
            )
            rate = rate_pct / 100.0
        else:
            annual_rate_pct = st.number_input(
                "Nominal Annual Rate (%)",
                value=6.0,
                step=0.1,
                format="%.4f",
                help=f"Converted to a per-period rate using {periods_per_year} periods per year.",
            )
            rate = (annual_rate_pct / 100.0) / periods_per_year

with col2:
    pmt_input = None
    nper_input = None

    if solve_for != "Payment per Period (PMT)":
        pmt_input = st.number_input(
            "Payment per Period (PMT)",
            value=0.0,
            step=100.0,
            format="%.2f",
            help="Use negative for outgoing payments",
        )

    if solve_for != "Number of Periods (NPER)":
        period_entry_mode = st.radio(
            "Period Input Type",
            ["Total Periods", "Years"],
            horizontal=True,
        )
        if period_entry_mode == "Total Periods":
            nper_input = st.number_input(
                "Number of Periods (NPER)",
                value=10.0,
                min_value=0.1,
                step=1.0,
                format="%.4f",
            )
        else:
            years_input = st.number_input(
                "Years",
                value=10.0,
                min_value=0.01,
                step=0.25,
                format="%.4f",
                help=f"Converted using {periods_per_year} periods per year.",
            )
            nper_input = years_input * periods_per_year

    period_type = st.selectbox(
        "Payment Timing",
        ["End of Period (Ordinary)", "Beginning of Period (Annuity Due)"],
    )
    when = 0 if "End" in period_type else 1

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if st.button("Calculate", use_container_width=True):
    try:
        result = None
        annual_rate_equivalent = None
        solve_meta = SOLVE_OPTIONS[solve_for]

        if solve_for == "Present Value (PV)":
            result = pv(rate=rate, nper=nper_input, pmt=pmt_input, fv=fv_input, when=when)
        elif solve_for == "Future Value (FV)":
            result = fv(rate=rate, nper=nper_input, pmt=pmt_input, pv_value=pv_input, when=when)
        elif solve_for == "Payment per Period (PMT)":
            result = pmt(rate=rate, nper=nper_input, pv_value=pv_input, fv_value=fv_input, when=when)
        elif solve_for == "Number of Periods (NPER)":
            result = nper(rate=rate, pmt_value=pmt_input, pv_value=pv_input, fv_value=fv_input, when=when)
        elif solve_for == "Interest Rate (Rate)":
            period_rate = rate_bisection(
                nper_value=nper_input,
                pmt_value=pmt_input,
                pv_value=pv_input,
                fv_value=fv_input,
                when=when,
            )
            result = period_rate * 100.0
            annual_rate_equivalent = period_rate * periods_per_year * 100.0

        if safe_is_valid(result):
            formatted = format_result(
                result,
                prefix=solve_meta["prefix"],
                suffix=solve_meta["suffix"],
            )
            st.markdown(
                f"""
            <div class='result-box'>
                <div class='result-label'>{solve_meta["label"]}</div>
                <div class='result-value'>{formatted}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            details = [
                f"Frequency: {period_frequency_label}",
                f"Payments: {'Beginning of period' if when == 1 else 'End of period'}",
            ]
            if solve_for != "Interest Rate (Rate)" and safe_is_valid(rate):
                details.append(f"Rate per period: {rate * 100:,.4f}%")
                details.append(f"Nominal annual rate: {rate * periods_per_year * 100:,.4f}%")
            if solve_for != "Number of Periods (NPER)" and safe_is_valid(nper_input):
                details.append(f"Total periods: {nper_input:,.4f}")
                details.append(f"Years: {nper_input / periods_per_year:,.4f}")
            if annual_rate_equivalent is not None:
                details.append(f"Nominal annual rate: {annual_rate_equivalent:,.4f}%")

            st.caption(" | ".join(details))
        else:
            st.error("Could not calculate a valid result. Please check your inputs.")

    except Exception as exc:
        st.error(f"Calculation error: {exc}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p class='info-text'>Tip: Use negative values for outgoing cash flows such as loan payments or investments. The calculator can solve for PV, FV, rate, PMT, or NPER.</p>",
    unsafe_allow_html=True,
)
