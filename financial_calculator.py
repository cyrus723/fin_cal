import math
import re

import numpy as np
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

CASH_FLOW_OPTIONS = {
    "Net Present Value (NPV)": {"label": "Net Present Value", "prefix": "$", "suffix": ""},
    "Internal Rate of Return (IRR)": {"label": "Internal Rate(s) of Return", "prefix": "", "suffix": "%"},
    "Modified Internal Rate of Return (MIRR)": {"label": "Modified Internal Rate of Return", "prefix": "", "suffix": "%"},
    "Profitability Index (PI)": {"label": "Profitability Index", "prefix": "", "suffix": ""},
}

SAMPLE_CASH_FLOWS = {
    "Custom": None,
    "Conventional Project": "-10000, 3000, 4200, 6800",
    "Multiple IRRs Example": "-100, 230, -132",
    "Five-Period Expansion": "-50000, 12000, 15000, 18000, 20000, 22000",
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


def parse_cash_flows(raw_text):
    tokens = [token for token in re.split(r"[\s,;]+", raw_text.strip()) if token]
    if not tokens:
        raise ValueError("Enter at least one cash flow.")

    cash_flows = []
    for token in tokens:
        try:
            cash_flows.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid cash flow value: {token}") from exc

    return cash_flows


def npv(rate, cash_flows):
    return sum(cash_flow / ((1 + rate) ** period) for period, cash_flow in enumerate(cash_flows))


def profitability_index(rate, cash_flows):
    initial_outlay = -sum(cash_flow for period, cash_flow in enumerate(cash_flows) if period == 0 and cash_flow < 0)
    if initial_outlay <= 0:
        raise ValueError("Profitability Index requires at least one negative cash flow at period 0.")

    discounted_future_inflows = sum(
        max(cash_flow, 0.0) / ((1 + rate) ** period)
        for period, cash_flow in enumerate(cash_flows)
        if period > 0
    )
    return discounted_future_inflows / initial_outlay


def irr_all(cash_flows, tol=1e-9):
    if len(cash_flows) < 2:
        raise ValueError("IRR requires at least two cash flows.")

    if not any(cash_flow < 0 for cash_flow in cash_flows) or not any(cash_flow > 0 for cash_flow in cash_flows):
        raise ValueError("IRR requires at least one negative and one positive cash flow.")

    normalized_cash_flows = list(cash_flows)
    while len(normalized_cash_flows) > 1 and abs(normalized_cash_flows[-1]) < tol:
        normalized_cash_flows.pop()

    # NPV(r) = c0 + c1*y + ... + cn*y^n where y = 1 / (1 + r).
    # numpy.roots expects descending powers, so reverse the cash-flow order.
    roots = np.roots(list(reversed(normalized_cash_flows)))
    rates = []

    for root in roots:
        if abs(root.imag) > 1e-7:
            continue
        real_root = root.real
        if real_root <= tol:
            continue
        rate = (1 / real_root) - 1
        if rate <= -1 + tol:
            continue
        if abs(npv(rate, normalized_cash_flows)) > 1e-5:
            continue
        rates.append(rate)

    unique_rates = []
    for rate in sorted(rates):
        if not unique_rates or abs(rate - unique_rates[-1]) > 1e-7:
            unique_rates.append(rate)

    if not unique_rates:
        raise ValueError("No real IRR values were found for this cash flow stream.")

    return unique_rates


def mirr(cash_flows, finance_rate, reinvestment_rate, tol=1e-9):
    if len(cash_flows) < 2:
        raise ValueError("MIRR requires at least two cash flows.")

    periods = len(cash_flows) - 1
    negative_cash_flows = [cash_flow for cash_flow in cash_flows if cash_flow < -tol]
    positive_cash_flows = [cash_flow for cash_flow in cash_flows if cash_flow > tol]

    if not negative_cash_flows or not positive_cash_flows:
        raise ValueError("MIRR requires at least one negative and one positive cash flow.")

    present_value_negatives = sum(
        cash_flow / ((1 + finance_rate) ** period)
        for period, cash_flow in enumerate(cash_flows)
        if cash_flow < -tol
    )
    future_value_positives = sum(
        cash_flow * ((1 + reinvestment_rate) ** (periods - period))
        for period, cash_flow in enumerate(cash_flows)
        if cash_flow > tol
    )

    if present_value_negatives >= 0 or future_value_positives <= 0:
        raise ValueError("Could not compute MIRR from these cash flows and rates.")

    return ((future_value_positives / -present_value_negatives) ** (1 / periods)) - 1


st.markdown("<h1 style='color:#ffffff; margin-bottom:4px;'>Financial Calculator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#7eb8f7; margin-bottom:28px;'>Time Value of Money and Cash Flow Analysis</p>",
    unsafe_allow_html=True,
)

calculator_mode = st.selectbox("Calculator Mode", ["Time Value of Money", "Cash Flow Analysis"])

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if calculator_mode == "Time Value of Money":
    solve_for = st.selectbox("Solve For", list(SOLVE_OPTIONS.keys()))

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
else:
    solve_for = st.selectbox("Cash Flow Metric", list(CASH_FLOW_OPTIONS.keys()))

    col1, col2 = st.columns(2)

    with col1:
        period_frequency_label = st.selectbox(
            "Cash Flow Frequency",
            list(FREQUENCY_OPTIONS.keys()),
            index=0,
        )
        periods_per_year = FREQUENCY_OPTIONS[period_frequency_label]

        discount_rate = None
        finance_rate = None
        reinvestment_rate = None
        if solve_for in {"Net Present Value (NPV)", "Profitability Index (PI)"}:
            rate_mode = st.radio(
                "Discount Rate Input Type",
                ["Rate per Period", "Nominal Annual Rate"],
                horizontal=True,
            )
            if rate_mode == "Rate per Period":
                discount_rate_pct = st.number_input(
                    "Discount Rate per Period (%)",
                    value=10.0,
                    step=0.1,
                    format="%.4f",
                )
                discount_rate = discount_rate_pct / 100.0
            else:
                annual_discount_rate_pct = st.number_input(
                    "Nominal Annual Discount Rate (%)",
                    value=12.0,
                    step=0.1,
                    format="%.4f",
                    help=f"Converted to a per-period rate using {periods_per_year} periods per year.",
                )
                discount_rate = (annual_discount_rate_pct / 100.0) / periods_per_year
        elif solve_for == "Modified Internal Rate of Return (MIRR)":
            mirr_rate_mode = st.radio(
                "MIRR Rate Input Type",
                ["Rate per Period", "Nominal Annual Rate"],
                horizontal=True,
            )
            if mirr_rate_mode == "Rate per Period":
                finance_rate_pct = st.number_input(
                    "Finance Rate per Period (%)",
                    value=8.0,
                    step=0.1,
                    format="%.4f",
                )
                reinvestment_rate_pct = st.number_input(
                    "Reinvestment Rate per Period (%)",
                    value=10.0,
                    step=0.1,
                    format="%.4f",
                )
                finance_rate = finance_rate_pct / 100.0
                reinvestment_rate = reinvestment_rate_pct / 100.0
            else:
                annual_finance_rate_pct = st.number_input(
                    "Nominal Annual Finance Rate (%)",
                    value=8.0,
                    step=0.1,
                    format="%.4f",
                    help=f"Converted to a per-period finance rate using {periods_per_year} periods per year.",
                )
                annual_reinvestment_rate_pct = st.number_input(
                    "Nominal Annual Reinvestment Rate (%)",
                    value=10.0,
                    step=0.1,
                    format="%.4f",
                    help=f"Converted to a per-period reinvestment rate using {periods_per_year} periods per year.",
                )
                finance_rate = (annual_finance_rate_pct / 100.0) / periods_per_year
                reinvestment_rate = (annual_reinvestment_rate_pct / 100.0) / periods_per_year

    with col2:
        sample_choice = st.selectbox("Sample Cash Flows", list(SAMPLE_CASH_FLOWS.keys()))
        if st.button("Load Sample", use_container_width=True, disabled=sample_choice == "Custom"):
            st.session_state["cash_flow_text"] = SAMPLE_CASH_FLOWS[sample_choice]

        if "cash_flow_text" not in st.session_state:
            st.session_state["cash_flow_text"] = "-10000, 3000, 4200, 6800"

        cash_flow_text = st.text_area(
            "Cash Flow Stream",
            key="cash_flow_text",
            height=180,
            help="Enter cash flows in chronological order. Separate values with commas, spaces, semicolons, or new lines.",
        )
        st.caption("Use a sample to test conventional projects and multiple-IRR streams quickly.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("Analyze Cash Flows", use_container_width=True):
        try:
            cash_flows = parse_cash_flows(cash_flow_text)
            solve_meta = CASH_FLOW_OPTIONS[solve_for]

            if solve_for == "Net Present Value (NPV)":
                result = npv(discount_rate, cash_flows)
                st.markdown(
                    f"""
                <div class='result-box'>
                    <div class='result-label'>{solve_meta["label"]}</div>
                    <div class='result-value'>{format_result(result, prefix='$')}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    " | ".join(
                        [
                            f"Cash flow frequency: {period_frequency_label}",
                            f"Discount rate per period: {discount_rate * 100:,.4f}%",
                            f"Nominal annual discount rate: {discount_rate * periods_per_year * 100:,.4f}%",
                            f"Periods analyzed: {len(cash_flows) - 1}",
                        ]
                    )
                )
            elif solve_for == "Modified Internal Rate of Return (MIRR)":
                result = mirr(cash_flows, finance_rate, reinvestment_rate)
                st.markdown(
                    f"""
                <div class='result-box'>
                    <div class='result-label'>{solve_meta["label"]}</div>
                    <div class='result-value'>{result * 100:,.4f}%</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    " | ".join(
                        [
                            f"Cash flow frequency: {period_frequency_label}",
                            f"Finance rate per period: {finance_rate * 100:,.4f}%",
                            f"Reinvestment rate per period: {reinvestment_rate * 100:,.4f}%",
                            f"Nominal annual finance rate: {finance_rate * periods_per_year * 100:,.4f}%",
                            f"Nominal annual reinvestment rate: {reinvestment_rate * periods_per_year * 100:,.4f}%",
                        ]
                    )
                )
            elif solve_for == "Profitability Index (PI)":
                result = profitability_index(discount_rate, cash_flows)
                st.markdown(
                    f"""
                <div class='result-box'>
                    <div class='result-label'>{solve_meta["label"]}</div>
                    <div class='result-value'>{result:,.4f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    " | ".join(
                        [
                            "PI = discounted future inflows / initial outlay",
                            f"Discount rate per period: {discount_rate * 100:,.4f}%",
                            f"Nominal annual discount rate: {discount_rate * periods_per_year * 100:,.4f}%",
                        ]
                    )
                )
            elif solve_for == "Internal Rate of Return (IRR)":
                irr_values = irr_all(cash_flows)
                formatted_rates = "<br>".join(
                    f"IRR {index}: {irr_value * 100:,.4f}%"
                    for index, irr_value in enumerate(irr_values, start=1)
                )
                st.markdown(
                    f"""
                <div class='result-box'>
                    <div class='result-label'>{solve_meta["label"]}</div>
                    <div class='result-value' style='font-size:28px;'>{formatted_rates}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                details = [f"Cash flow frequency: {period_frequency_label}", f"Real IRR solutions found: {len(irr_values)}"]
                details.extend(
                    f"IRR {index} nominal annual equivalent: {irr_value * periods_per_year * 100:,.4f}%"
                    for index, irr_value in enumerate(irr_values, start=1)
                )
                st.caption(" | ".join(details))

                if len(irr_values) > 1:
                    st.warning(
                        "This cash flow stream has multiple valid IRRs. Use NPV or MIRR-style decision rules if you need a single ranking metric."
                    )
        except Exception as exc:
            st.error(f"Calculation error: {exc}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p class='info-text'>Tip: Use negative values for outgoing cash flows such as loan payments or investments. The calculator can solve TVM metrics plus NPV, IRR, MIRR, and PI from ordered cash flow streams.</p>",
    unsafe_allow_html=True,
)
