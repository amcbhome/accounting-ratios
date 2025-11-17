import io
from typing import Dict, Any, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Demo dataset: Stationery & Computer Mart UK
# -----------------------------
def load_demo_trial_balance() -> pd.DataFrame:
    """
    Hard-coded demo TB based on the Sage 50 Stationery & Computer Mart
    transactional trial balance for 01/01/2024–01/12/2024.

    Columns:
        code   : nominal code (int)
        name   : account name
        debit  : debit balance in period
        credit : credit balance in period
    """
    data = [
        (21,   "Plant/Machinery Depreciation",            515.00,    0.00),
        (51,   "Motor Vehicles Depreciation",             757.44,    0.00),
        (1100, "Debtors Control Account",              89731.16,    0.00),
        (1103, "Prepayments",                            1350.00,    0.00),
        (1200, "Bank Current Account",                   3389.99,    0.00),
        (1210, "Bank Deposit Account",                   2000.00,    0.00),
        (1220, "Building Society Account",                505.03,    0.00),
        (1230, "Petty Cash",                              833.48,    0.00),
        (1240, "Company Credit Card",                       0.00, 10414.97),
        (2100, "Creditors Control Account",                 0.00, 36572.97),
        (2109, "Accruals",                                  0.00,    50.00),
        (2200, "Sales Tax Control Account",                 0.00, 22152.44),
        (2201, "Purchase Tax Control Account",              0.00, 11102.51),
        (2202, "VAT Liability",                             0.00, 14800.35),
        (2210, "P.A.Y.E.",                                  0.00,  2070.23),
        (2211, "National Insurance",                        0.00,  1003.49),
        (2220, "Net Wages",                                 0.00,     0.00),
        (2230, "Pension Fund",                              0.00,    80.00),
        (2300, "Loans",                                     0.00,   605.00),
        (2310, "Hire Purchase",                             0.00,  1800.00),
        (4000, "Sales North",                               0.00, 179507.53),
        (4001, "Sales South",                               0.00,   1230.00),
        (4002, "Sales Scotland",                            0.00,   8472.51),
        (4009, "Discounts Allowed",                         50.00,     0.00),
        (4900, "Miscellaneous Income",                      0.00,    60.03),
        (4905, "Distribution and Carriage",                870.00,     0.00),
        (5000, "Materials Purchased",                     45446.48,    0.00),
        (5001, "Materials Imported",                      23733.00,    0.00),
        (5002, "Miscellaneous Purchases",                 1158.53,     0.00),
        (5100, "Carriage",                                   1.26,     0.00),
        (6200, "Sales Promotions",                          50.00,     0.00),
        (6201, "Advertising",                              465.00,     0.00),
        (6202, "Gifts and Samples",                         15.00,     0.00),
        (6203, "P.R. (Literature & Brochures)",           1050.00,     0.00),
        (7000, "Gross Wages",                            24372.11,    0.00),
        (7006, "Employers N.I.",                          2495.43,    0.00),
        (7009, "Adjustments",                              170.00,    0.00),
        (7010, "SSP Reclaimed",                              0.00,    30.00),
        (7011, "SMP Reclaimed",                              0.00,    48.40),
        (7100, "Rent",                                   15750.00,     0.00),
        (7200, "Electricity",                               952.00,   0.00),
        (7300, "Fuel and Oil",                               15.00,   0.00),
        (7301, "Repairs and Servicing",                      88.18,   0.00),
        (7304, "Miscellaneous Motor Expenses",               67.00,   0.00),
        (7350, "Scale Charges",                              60.18,   0.00),
        (7400, "Travelling",                                201.00,   0.00),
        (7401, "Car Hire",                                  150.00,   0.00),
        (7402, "Hotels",                                    720.00,   0.00),
        (7403, "U.K. Entertainment",                          5.50,   0.00),
        (7500, "Printing",                                   51.60,   0.00),
        (7501, "Postage and Carriage",                        3.50,   0.00),
        (7502, "Telephone",                                 128.72,   0.00),
        (7802, "Laundry",                                    50.00,   0.00),
        (7901, "Bank Charges",                                5.56,   0.00),
        (7903, "Loan Interest Paid",                         83.25,   0.00),
        (8003, "Vehicle Depreciation",                      757.44,   0.00),
        (8100, "Bad Debt Write Off",                          0.01,   0.00),
        (9999, "Mispostings Account",                       205.00,   0.00),
    ]
    df = pd.DataFrame(data, columns=["code", "name", "debit", "credit"])
    df["code"] = df["code"].astype(int)
    df["balance"] = df["debit"] - df["credit"]
    return df


# -----------------------------
# Utility functions
# -----------------------------
def normalise_tb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to standardise an uploaded Sage TB into columns: code, name, debit, credit.

    Looks for column names containing 'code', 'name', 'debit', 'credit'.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    def find_col(keyword: str):
        for c in df.columns:
            if keyword in c.lower():
                return c
        return None

    code_col = find_col("code") or find_col("n/c") or find_col("nominal")
    name_col = find_col("name")
    debit_col = find_col("debit")
    credit_col = find_col("credit")

    missing = [x for x in [code_col, name_col, debit_col, credit_col] if x is None]
    if missing:
        raise ValueError(
            "Could not automatically detect columns for code/name/debit/credit. "
            "Please ensure your export includes those headings."
        )

    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str).str.extract(r"(\d+)")[0].astype(int)
    out["name"] = df[name_col].astype(str)
    out["debit"] = pd.to_numeric(df[debit_col], errors="coerce").fillna(0.0)
    out["credit"] = pd.to_numeric(df[credit_col], errors="coerce").fillna(0.0)
    out["balance"] = out["debit"] - out["credit"]
    return out


def calculate_ratios(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Calculate a set of ratios from a standardised trial balance DataFrame.

    The logic is tailored to the Sage 50 demo chart of accounts but is easy
    to adapt to other nominal code structures.
    """
    def bal(code: int) -> float:
        row = df[df["code"] == code]
        return (row["balance"]).sum()

    # --- Revenue & cost of sales ---
    sales_codes = [4000, 4001, 4002]
    sales = -df[df["code"].isin(sales_codes)]["balance"].sum()  # credits -> positive
    discounts = df[df["code"] == 4009]["balance"].sum()
    net_sales = sales - discounts

    cogs_codes = [5000, 5001, 5002, 5100]
    cogs = df[df["code"].isin(cogs_codes)]["balance"].sum()

    gross_profit = net_sales - cogs
    gross_margin_pct = (gross_profit / net_sales * 100) if net_sales else None

    # --- Operating expenses ---
    expense_codes = [
        4905, 6200, 6201, 6202, 6203,
        7000, 7006, 7009,
        7100, 7200,
        7300, 7301, 7304,
        7350, 7400, 7401, 7402, 7403,
        7500, 7501, 7502,
        7802,
        7901,
        8003,
        8100,
    ]
    expenses = df[df["code"].isin(expense_codes)]["balance"].sum()
    # SSP/SMP reclaimed – treated as reductions in wages
    ssp_smp_recovery = -df[df["code"].isin([7010, 7011])]["balance"].sum()
    operating_expenses = expenses - ssp_smp_recovery

    misc_income = -bal(4900)  # credit balance -> positive income

    operating_profit = gross_profit - operating_expenses + misc_income
    operating_margin_pct = (operating_profit / net_sales * 100) if net_sales else None

    # --- Finance cost & profit before tax ---
    finance_cost = df[df["code"] == 7903]["balance"].sum()
    profit_before_tax = operating_profit - finance_cost
    net_margin_pct = (profit_before_tax / net_sales * 100) if net_sales else None

    # --- Working capital & liquidity ---
    current_asset_codes = [1100, 1103, 1200, 1210, 1220, 1230]
    current_liability_codes = [2100, 2109, 2200, 2201, 2202, 2210, 2211, 2220, 2230, 1240]

    current_assets = df[df["code"].isin(current_asset_codes)]["balance"].sum()
    # liabilities are credit balances (negative), so flip sign
    current_liabilities = -df[df["code"].isin(current_liability_codes)]["balance"].sum()
    working_capital = current_assets - current_liabilities

    current_ratio = (current_assets / current_liabilities) if current_liabilities else None
    quick_assets = current_assets - df[df["code"] == 1103]["balance"].sum()  # exclude prepayments
    quick_ratio = (quick_assets / current_liabilities) if current_liabilities else None

    # --- Efficiency ratios ---
    debtors = bal(1100)
    creditors = -bal(2100)  # credit balance
    receivables_days = (debtors / net_sales * 365) if net_sales else None
    payables_days = (creditors / cogs * 365) if cogs else None

    ratios = {
        "Net sales (£)": net_sales,
        "Gross profit (£)": gross_profit,
        "Gross margin (%)": gross_margin_pct,
        "Operating profit (£)": operating_profit,
        "Operating margin (%)": operating_margin_pct,
        "Profit before tax (£)": profit_before_tax,
        "Net margin (PBT, %)": net_margin_pct,
        "Current assets (£)": current_assets,
        "Current liabilities (£)": current_liabilities,
        "Working capital (£)": working_capital,
        "Current ratio": current_ratio,
        "Quick ratio": quick_ratio,
        "Receivables days": receivables_days,
        "Payables days": payables_days,
    }

    # Build a small breakdown table for the dashboard chart
    breakdown = pd.DataFrame(
        {
            "Category": [
                "Net sales",
                "Cost of sales",
                "Gross profit",
                "Operating expenses",
                "Operating profit",
                "Finance cost",
                "Profit before tax",
            ],
            "Amount": [
                net_sales,
                -cogs,  # show as negative bar
                gross_profit,
                -operating_expenses,
                operating_profit,
                -finance_cost,
                profit_before_tax,
            ],
        }
    )

    return ratios, breakdown


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Trial Balance Ratio Dashboard",
        layout="wide",
    )

    st.title("Accounting Ratio Dashboard")
    st.caption("Sage 50 transactional trial balance → Key ratios & visuals")

    st.sidebar.header("1. Data source")

    use_demo = st.sidebar.toggle(
        "Use demo Stationery & Computer Mart TB",
        value=True,
        help="If ticked, uses the built-in Sage 50 demo data. "
             "Untick to upload your own trial balance (CSV/Excel).",
    )

    uploaded_file = None
    df_tb = None

    if not use_demo:
        uploaded_file = st.sidebar.file_uploader(
            "Upload trial balance (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Export the transactional trial balance from Sage and upload it here.",
        )

    if use_demo:
        df_tb = load_demo_trial_balance()
        st.sidebar.success("Using built-in Sage 50 demo trial balance.")
    elif uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                raw = pd.read_csv(uploaded_file)
            else:
                raw = pd.read_excel(uploaded_file)

            df_tb = normalise_tb_columns(raw)
            st.sidebar.success("File uploaded and parsed successfully.")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

    if df_tb is None:
        st.info("Upload a trial balance file or enable the demo data to view ratios.")
        return

    # Show raw TB in an expander
    with st.expander("View trial balance data"):
        st.dataframe(df_tb, use_container_width=True)

    # Calculate ratios
    ratios, breakdown = calculate_ratios(df_tb)

    # -----------------------------
    # KPI tiles
    # -----------------------------
    st.subheader("Key profitability ratios")

    col1, col2, col3 = st.columns(3)
    col1.metric("Net sales (£)", f"{ratios['Net sales (£)']:,.2f}")
    col2.metric("Gross profit (£)", f"{ratios['Gross profit (£)']:,.2f}",
                f"{ratios['Gross margin (%)']:.2f}% GP margin")
    col3.metric("Operating profit (£)", f"{ratios['Operating profit (£)']:,.2f}",
                f"{ratios['Operating margin (%)']:.2f}% OP margin")

    col4, col5 = st.columns(2)
    col4.metric("Profit before tax (£)", f"{ratios['Profit before tax (£)']:,.2f}",
                f"{ratios['Net margin (PBT, %)']:.2f}% net margin")
    col5.metric("Finance cost (£)", f"{-breakdown.loc[breakdown['Category']=='Finance cost','Amount'].iloc[0]:,.2f}")

    st.subheader("Liquidity & working capital")

    col6, col7, col8 = st.columns(3)
    col6.metric("Current ratio", f"{ratios['Current ratio']:.2f}")
    col7.metric("Quick ratio", f"{ratios['Quick ratio']:.2f}")
    col8.metric("Working capital (£)", f"{ratios['Working capital (£)']:,.2f}")

    col9, col10 = st.columns(2)
    col9.metric("Receivables days", f"{ratios['Receivables days']:.1f} days")
    col10.metric("Payables days", f"{ratios['Payables days']:.1f} days")

    # -----------------------------
    # Visuals
    # -----------------------------
    st.subheader("P&L structure")
    st.bar_chart(
        breakdown.set_index("Category")["Amount"],
        use_container_width=True,
    )

    st.markdown(
        """
        **Notes & assumptions**

        - Mapping from nominal codes to P&L and working capital categories is based on the
          default Sage 50 UK chart of accounts for the demo company.
        - SSP/SMP reclaimed are treated as reductions in staff costs.
        - The transactional trial balance does not include all balance sheet headings
          (capital, fixed assets at cost, inventory etc.), so ROCE/gearing are not shown.
        """
    )


if __name__ == "__main__":
    main()
