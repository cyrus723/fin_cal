import math
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

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

BOND_DURATION_OPTIONS = {
    "Macaulay Duration": {"label": "Macaulay Duration", "prefix": "", "suffix": " years"},
    "Modified Duration": {"label": "Modified Duration", "prefix": "", "suffix": ""},
    "Dollar Duration": {"label": "Dollar Duration", "prefix": "$", "suffix": ""},
    "DV01": {"label": "DV01", "prefix": "$", "suffix": ""},
    "Convexity": {"label": "Convexity", "prefix": "", "suffix": ""},
}

SAMPLE_CASH_FLOWS = {
    "Custom": None,
    "Conventional Project": "-10000, 3000, 4200, 6800",
    "Multiple IRRs Example": "-100, 230, -132",
    "Five-Period Expansion": "-50000, 12000, 15000, 18000, 20000, 22000",
}

PORTFOLIO_LOOKBACK_OPTIONS = {
    "6 Months": "6mo",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y",
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


def bond_cash_flows(face_value, coupon_rate, years_to_maturity, payments_per_year):
    total_periods = int(round(years_to_maturity * payments_per_year))
    if total_periods <= 0:
        raise ValueError("Bond maturity must produce at least one payment period.")

    coupon_payment = face_value * coupon_rate / payments_per_year
    cash_flows = [coupon_payment] * total_periods
    cash_flows[-1] += face_value
    return cash_flows


def bond_price(face_value, coupon_rate, yield_rate, years_to_maturity, payments_per_year):
    cash_flows = bond_cash_flows(face_value, coupon_rate, years_to_maturity, payments_per_year)
    period_yield = yield_rate / payments_per_year
    return sum(
        cash_flow / ((1 + period_yield) ** period)
        for period, cash_flow in enumerate(cash_flows, start=1)
    )


def bond_durations(face_value, coupon_rate, yield_rate, years_to_maturity, payments_per_year):
    if payments_per_year <= 0:
        raise ValueError("Payments per year must be greater than 0.")
    if years_to_maturity <= 0:
        raise ValueError("Years to maturity must be greater than 0.")
    if yield_rate <= -payments_per_year:
        raise ValueError("Yield is too low for a valid discount rate.")

    cash_flows = bond_cash_flows(face_value, coupon_rate, years_to_maturity, payments_per_year)
    period_yield = yield_rate / payments_per_year
    price = bond_price(face_value, coupon_rate, yield_rate, years_to_maturity, payments_per_year)

    if price <= 0:
        raise ValueError("Bond price must be positive to compute duration.")

    weighted_present_values = 0.0
    convexity_numerator = 0.0
    for period, cash_flow in enumerate(cash_flows, start=1):
        time_years = period / payments_per_year
        present_value = cash_flow / ((1 + period_yield) ** period)
        weighted_present_values += time_years * present_value
        convexity_numerator += cash_flow * period * (period + 1) / ((1 + period_yield) ** (period + 2))

    macaulay_duration = weighted_present_values / price
    modified_duration = macaulay_duration / (1 + period_yield)
    dollar_duration = modified_duration * price
    dv01 = dollar_duration * 0.0001
    convexity = convexity_numerator / (price * (payments_per_year**2))

    return {
        "price": price,
        "macaulay_duration": macaulay_duration,
        "modified_duration": modified_duration,
        "dollar_duration": dollar_duration,
        "dv01": dv01,
        "convexity": convexity,
        "coupon_payment": face_value * coupon_rate / payments_per_year,
        "total_periods": len(cash_flows),
        "period_yield": period_yield,
    }


def parse_symbols(symbol_inputs):
    symbols = []
    for raw_symbol in symbol_inputs:
        normalized = raw_symbol.strip().upper()
        if not normalized:
            continue
        if normalized in symbols:
            raise ValueError(f"Duplicate symbol entered: {normalized}")
        symbols.append(normalized)

    if not symbols:
        raise ValueError("Enter at least one stock symbol.")
    if len(symbols) > 5:
        raise ValueError("Portfolio analysis supports up to five stock symbols.")

    return symbols


@st.cache_data(show_spinner=False)
def fetch_price_history(symbols, lookback_period):
    price_frames = []

    for symbol in symbols:
        history = yf.download(
            symbol,
            period=lookback_period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if history.empty or "Close" not in history.columns:
            raise ValueError(f"No price history was returned for {symbol}.")

        closes = history[["Close"]].rename(columns={"Close": symbol})
        price_frames.append(closes)

    prices = pd.concat(price_frames, axis=1, join="inner").dropna()
    if prices.shape[0] < 3:
        raise ValueError("Not enough overlapping price history was available for these symbols.")

    return prices


def portfolio_metrics(weights, annual_returns, annual_covariance):
    portfolio_return = float(weights @ annual_returns)
    portfolio_variance = float(weights @ annual_covariance @ weights)
    portfolio_std_dev = math.sqrt(max(portfolio_variance, 0.0))
    return portfolio_return, portfolio_std_dev


def efficient_frontier(annual_returns, annual_covariance, num_points=80):
    if len(annual_returns) == 1:
        single_return = float(annual_returns[0])
        single_std_dev = math.sqrt(max(float(annual_covariance[0][0]), 0.0))
        return (
            [{"return": single_return, "volatility": single_std_dev}],
            np.array([1.0]),
            single_return,
            single_std_dev,
        )

    ones = np.ones(len(annual_returns))
    inverse_covariance = np.linalg.pinv(annual_covariance)

    a_value = float(ones @ inverse_covariance @ ones)
    b_value = float(ones @ inverse_covariance @ annual_returns)
    c_value = float(annual_returns @ inverse_covariance @ annual_returns)
    delta = a_value * c_value - b_value**2

    if a_value <= 0 or delta <= 1e-12:
        raise ValueError("Could not build an efficient frontier from this covariance matrix.")

    global_minimum_return = b_value / a_value
    target_returns = np.linspace(global_minimum_return, max(float(np.max(annual_returns)), global_minimum_return), num_points)

    frontier_points = []
    for target_return in target_returns:
        variance = (a_value * target_return**2 - 2 * b_value * target_return + c_value) / delta
        frontier_points.append(
            {
                "return": target_return,
                "volatility": math.sqrt(max(variance, 0.0)),
            }
        )

    gmvp_weights = (inverse_covariance @ ones) / a_value
    gmvp_return, gmvp_std_dev = portfolio_metrics(gmvp_weights, annual_returns, annual_covariance)

    return frontier_points, gmvp_weights, gmvp_return, gmvp_std_dev


def tangency_portfolio(annual_returns, annual_covariance, risk_free_rate):
    excess_returns = annual_returns - risk_free_rate
    inverse_covariance = np.linalg.pinv(annual_covariance)
    raw_weights = inverse_covariance @ excess_returns
    normalization = float(np.sum(raw_weights))

    if abs(normalization) <= 1e-12:
        raise ValueError("Could not compute an optimal portfolio from these inputs.")

    optimal_weights = raw_weights / normalization
    optimal_return, optimal_std_dev = portfolio_metrics(optimal_weights, annual_returns, annual_covariance)
    sharpe_ratio = (
        (optimal_return - risk_free_rate) / optimal_std_dev
        if optimal_std_dev > 0
        else 0.0
    )
    return optimal_weights, optimal_return, optimal_std_dev, sharpe_ratio


st.markdown("<h1 style='color:#ffffff; margin-bottom:4px;'>Financial Calculator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#7eb8f7; margin-bottom:28px;'>Time Value of Money, Cash Flow Analysis, Bond Duration, and Portfolio Analytics</p>",
    unsafe_allow_html=True,
)

calculator_mode = st.selectbox(
    "Calculator Mode",
    ["Time Value of Money", "Cash Flow Analysis", "Bond Duration", "Portfolio Analytics"],
)

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
elif calculator_mode == "Cash Flow Analysis":
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
elif calculator_mode == "Bond Duration":
    solve_for = st.selectbox("Bond Metric", list(BOND_DURATION_OPTIONS.keys()))

    col1, col2 = st.columns(2)

    with col1:
        face_value = st.number_input(
            "Face Value",
            value=1000.0,
            min_value=0.01,
            step=100.0,
            format="%.2f",
        )
        coupon_rate_pct = st.number_input(
            "Annual Coupon Rate (%)",
            value=5.0,
            min_value=0.0,
            step=0.1,
            format="%.4f",
        )
        years_to_maturity = st.number_input(
            "Years to Maturity",
            value=10.0,
            min_value=0.01,
            step=0.25,
            format="%.4f",
        )

    with col2:
        payment_frequency_label = st.selectbox(
            "Coupon Frequency",
            ["Annual", "Semiannual", "Quarterly", "Monthly"],
            index=1,
        )
        payments_per_year = FREQUENCY_OPTIONS[payment_frequency_label]

        yield_mode = st.radio(
            "Yield Input Type",
            ["Nominal Annual Yield", "Yield per Period"],
            horizontal=True,
        )
        if yield_mode == "Nominal Annual Yield":
            annual_yield_pct = st.number_input(
                "Yield to Maturity (%)",
                value=4.5,
                step=0.1,
                format="%.4f",
            )
            yield_rate = annual_yield_pct / 100.0
        else:
            period_yield_pct = st.number_input(
                "Yield per Period (%)",
                value=2.25,
                step=0.1,
                format="%.4f",
                help=f"Converted to nominal annual yield using {payments_per_year} periods per year.",
            )
            yield_rate = (period_yield_pct / 100.0) * payments_per_year

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("Analyze Bond", use_container_width=True):
        try:
            bond_metrics = bond_durations(
                face_value=face_value,
                coupon_rate=coupon_rate_pct / 100.0,
                yield_rate=yield_rate,
                years_to_maturity=years_to_maturity,
                payments_per_year=payments_per_year,
            )
            solve_meta = BOND_DURATION_OPTIONS[solve_for]

            result_map = {
                "Macaulay Duration": bond_metrics["macaulay_duration"],
                "Modified Duration": bond_metrics["modified_duration"],
                "Dollar Duration": bond_metrics["dollar_duration"],
                "DV01": bond_metrics["dv01"],
                "Convexity": bond_metrics["convexity"],
            }
            result = result_map[solve_for]

            st.markdown(
                f"""
            <div class='result-box'>
                <div class='result-label'>{solve_meta["label"]}</div>
                <div class='result-value'>{format_result(result, prefix=solve_meta["prefix"], suffix=solve_meta["suffix"])}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.caption(
                " | ".join(
                    [
                        f"Bond price: {format_result(bond_metrics['price'], prefix='$')}",
                        f"Coupon payment: {format_result(bond_metrics['coupon_payment'], prefix='$')}",
                        f"Coupon frequency: {payment_frequency_label}",
                        f"Total coupon periods: {bond_metrics['total_periods']}",
                        f"Yield per period: {bond_metrics['period_yield'] * 100:,.4f}%",
                        f"Nominal annual yield: {yield_rate * 100:,.4f}%",
                        f"Modified duration: {bond_metrics['modified_duration']:,.4f}",
                        f"Convexity: {bond_metrics['convexity']:,.4f}",
                        f"DV01: {format_result(bond_metrics['dv01'], prefix='$')}",
                    ]
                )
            )
        except Exception as exc:
            st.error(f"Calculation error: {exc}")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Enter up to five stock symbols. Blank fields are ignored.")
        default_symbols = ["AAPL", "MSFT", "GOOGL", "", ""]
        symbol_inputs = [
            st.text_input(f"Stock Symbol {index}", value=default_symbols[index - 1], max_chars=10)
            for index in range(1, 6)
        ]
        lookback_label = st.selectbox("Historical Price Window", list(PORTFOLIO_LOOKBACK_OPTIONS.keys()), index=1)
        lookback_period = PORTFOLIO_LOOKBACK_OPTIONS[lookback_label]

    with col2:
        try:
            active_symbols = parse_symbols(symbol_inputs)
        except ValueError:
            active_symbols = []

        risk_free_rate_pct = st.number_input(
            "Risk-Free Rate (%)",
            value=2.0,
            step=0.1,
            format="%.2f",
            help="Used to identify the maximum Sharpe ratio portfolio.",
        )

        st.caption("Portfolio weights are normalized automatically if they do not add to 100%.")
        default_weight = 100.0 / max(len(active_symbols), 1)
        raw_weights = []
        for index, symbol in enumerate(active_symbols, start=1):
            raw_weights.append(
                st.number_input(
                    f"{symbol} Weight (%)",
                    value=default_weight,
                    min_value=0.0,
                    step=1.0,
                    format="%.2f",
                    key=f"weight_{symbol}_{index}",
                )
            )

        if not active_symbols:
            st.info("Add at least one valid stock symbol to enter portfolio weights.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("Analyze Portfolio", use_container_width=True):
        try:
            symbols = parse_symbols(symbol_inputs)
            if not raw_weights or len(raw_weights) != len(symbols):
                raise ValueError("Enter portfolio weights for each selected symbol.")

            weight_vector = np.array(raw_weights, dtype=float)
            if np.any(weight_vector < 0):
                raise ValueError("Portfolio weights cannot be negative.")
            if float(np.sum(weight_vector)) <= 0:
                raise ValueError("Portfolio weights must sum to more than 0%.")

            normalized_weights = weight_vector / np.sum(weight_vector)
            prices = fetch_price_history(tuple(symbols), lookback_period)
            daily_returns = prices.pct_change().dropna()

            annual_returns = daily_returns.mean().to_numpy() * 252
            annual_covariance = daily_returns.cov().to_numpy() * 252
            portfolio_return, portfolio_std_dev = portfolio_metrics(
                normalized_weights,
                annual_returns,
                annual_covariance,
            )
            risk_free_rate = risk_free_rate_pct / 100.0
            frontier_points, gmvp_weights, gmvp_return, gmvp_std_dev = efficient_frontier(
                annual_returns,
                annual_covariance,
            )
            optimal_weights, optimal_return, optimal_std_dev, optimal_sharpe = tangency_portfolio(
                annual_returns,
                annual_covariance,
                risk_free_rate,
            )

            weights_table = pd.DataFrame(
                {
                    "Symbol": symbols,
                    "Weight (%)": normalized_weights * 100,
                    "Annualized Return (%)": annual_returns * 100,
                    "Annualized Volatility (%)": np.sqrt(np.diag(annual_covariance)) * 100,
                }
            )
            correlation_matrix = daily_returns.corr()
            frontier_df = pd.DataFrame(frontier_points).rename(
                columns={"return": "Expected Return", "volatility": "Volatility"}
            )
            asset_points_df = pd.DataFrame(
                {
                    "Label": symbols,
                    "Volatility": np.sqrt(np.diag(annual_covariance)) * 100,
                    "Expected Return": annual_returns * 100,
                    "Series": ["Stock"] * len(symbols),
                }
            )
            frontier_chart_data = frontier_df.copy()
            frontier_chart_data["Expected Return"] *= 100
            frontier_chart_data["Volatility"] *= 100
            frontier_chart_data["Label"] = "Efficient Frontier"
            frontier_chart_data["Series"] = "Efficient Frontier"

            portfolio_points_df = pd.DataFrame(
                [
                    {
                        "Label": "Your Portfolio",
                        "Volatility": portfolio_std_dev * 100,
                        "Expected Return": portfolio_return * 100,
                        "Series": "Portfolio",
                    },
                    {
                        "Label": "Optimal Portfolio",
                        "Volatility": optimal_std_dev * 100,
                        "Expected Return": optimal_return * 100,
                        "Series": "Optimal",
                    },
                    {
                        "Label": "GMV Portfolio",
                        "Volatility": gmvp_std_dev * 100,
                        "Expected Return": gmvp_return * 100,
                        "Series": "GMV",
                    },
                ]
            )

            st.markdown(
                f"""
            <div class='result-box'>
                <div class='result-label'>Portfolio Return / Standard Deviation</div>
                <div class='result-value' style='font-size:28px;'>
                    {portfolio_return * 100:,.2f}% / {portfolio_std_dev * 100:,.2f}%
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            date_start = prices.index.min().date()
            date_end = prices.index.max().date()
            st.caption(
                " | ".join(
                    [
                        f"Symbols: {', '.join(symbols)}",
                        f"History window: {lookback_label}",
                        f"Observations: {len(daily_returns)} daily returns",
                        f"Date range: {date_start} to {date_end}",
                        f"Risk-free rate: {risk_free_rate_pct:,.2f}%",
                        f"Global minimum variance return: {gmvp_return * 100:,.2f}%",
                        f"Global minimum variance volatility: {gmvp_std_dev * 100:,.2f}%",
                    ]
                )
            )

            st.subheader("Portfolio Weights and Asset Statistics")
            st.dataframe(
                weights_table.style.format(
                    {
                        "Weight (%)": "{:,.2f}",
                        "Annualized Return (%)": "{:,.2f}",
                        "Annualized Volatility (%)": "{:,.2f}",
                    }
                ),
                use_container_width=True,
            )

            st.subheader("Correlation Matrix")
            st.dataframe(correlation_matrix.style.format("{:,.4f}"), use_container_width=True)

            st.subheader("Efficient Frontier")
            st.vega_lite_chart(
                {
                    "layer": [
                        {
                            "data": {"values": frontier_chart_data.to_dict("records")},
                            "mark": {"type": "line", "color": "#7eb8f7", "strokeWidth": 3},
                            "encoding": {
                                "x": {"field": "Volatility", "type": "quantitative", "title": "Volatility (%)"},
                                "y": {
                                    "field": "Expected Return",
                                    "type": "quantitative",
                                    "title": "Expected Return (%)",
                                },
                            },
                        },
                        {
                            "data": {"values": asset_points_df.to_dict("records")},
                            "mark": {"type": "point", "filled": True, "size": 140, "color": "#f6ad55"},
                            "encoding": {
                                "x": {"field": "Volatility", "type": "quantitative"},
                                "y": {"field": "Expected Return", "type": "quantitative"},
                                "tooltip": [
                                    {"field": "Label", "type": "nominal", "title": "Asset"},
                                    {"field": "Expected Return", "type": "quantitative", "format": ".2f"},
                                    {"field": "Volatility", "type": "quantitative", "format": ".2f"},
                                ],
                            },
                        },
                        {
                            "data": {"values": asset_points_df.to_dict("records")},
                            "mark": {"type": "text", "dy": -12, "color": "#fbd38d", "fontSize": 12},
                            "encoding": {
                                "x": {"field": "Volatility", "type": "quantitative"},
                                "y": {"field": "Expected Return", "type": "quantitative"},
                                "text": {"field": "Label", "type": "nominal"},
                            },
                        },
                        {
                            "data": {"values": portfolio_points_df.to_dict("records")},
                            "mark": {"type": "point", "filled": True, "size": 220},
                            "encoding": {
                                "x": {"field": "Volatility", "type": "quantitative"},
                                "y": {"field": "Expected Return", "type": "quantitative"},
                                "color": {
                                    "field": "Series",
                                    "type": "nominal",
                                    "scale": {
                                        "domain": ["Portfolio", "Optimal", "GMV"],
                                        "range": ["#68d391", "#fc8181", "#63b3ed"],
                                    },
                                    "legend": {"title": ""},
                                },
                                "shape": {
                                    "field": "Series",
                                    "type": "nominal",
                                    "legend": None,
                                },
                                "tooltip": [
                                    {"field": "Label", "type": "nominal"},
                                    {"field": "Expected Return", "type": "quantitative", "format": ".2f"},
                                    {"field": "Volatility", "type": "quantitative", "format": ".2f"},
                                ],
                            },
                        },
                        {
                            "data": {"values": portfolio_points_df.to_dict("records")},
                            "mark": {"type": "text", "dx": 10, "dy": -10, "fontSize": 12, "color": "#ffffff"},
                            "encoding": {
                                "x": {"field": "Volatility", "type": "quantitative"},
                                "y": {"field": "Expected Return", "type": "quantitative"},
                                "text": {"field": "Label", "type": "nominal"},
                            },
                        },
                    ],
                    "config": {
                        "background": "#0f1117",
                        "axis": {"labelColor": "#e2e8f0", "titleColor": "#e2e8f0", "gridColor": "#2d3748"},
                        "legend": {"labelColor": "#e2e8f0", "titleColor": "#e2e8f0"},
                        "view": {"stroke": "#2d3748"},
                    },
                },
                use_container_width=True,
            )
            st.caption(
                "The frontier is an unconstrained Markowitz mean-variance frontier built from annualized historical returns and covariance. The optimal portfolio is the maximum Sharpe ratio portfolio."
            )

            gmvp_weights_percent = ", ".join(
                f"{symbol}: {weight * 100:,.2f}%"
                for symbol, weight in zip(symbols, gmvp_weights)
            )
            st.caption(f"Global minimum variance portfolio weights: {gmvp_weights_percent}")

            st.subheader("Optimal Portfolio Weights")
            optimal_weights_table = pd.DataFrame(
                {
                    "Symbol": symbols,
                    "Optimal Weight (%)": optimal_weights * 100,
                }
            )
            st.dataframe(
                optimal_weights_table.style.format({"Optimal Weight (%)": "{:,.2f}"}),
                use_container_width=True,
            )
            st.caption(
                " | ".join(
                    [
                        f"Optimal expected return: {optimal_return * 100:,.2f}%",
                        f"Optimal volatility: {optimal_std_dev * 100:,.2f}%",
                        f"Optimal Sharpe ratio: {optimal_sharpe:,.4f}",
                    ]
                )
            )
        except Exception as exc:
            st.error(f"Calculation error: {exc}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p class='info-text'>Tip: Use negative values for outgoing cash flows such as loan payments or investments. The calculator can solve TVM metrics, cash flow analytics, bond duration measures, and portfolio analytics from free historical stock price data.</p>",
    unsafe_allow_html=True,
)
