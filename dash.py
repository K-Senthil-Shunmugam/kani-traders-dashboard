import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
import json
import numpy as np
import sklearn
st.set_page_config(layout="wide")

# -------------------------------
# 🌿 Custom Blue-Green-Themed Styling
st.markdown("""
    <style>
    /* --- Main Page Background --- */
    html, body, .main {
        background-color: #dceefb; /* Slightly deeper light blue */
        color: #0d1b2a;
        font-family: 'Segoe UI', sans-serif;
     
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #f5fbff;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    }

    /* --- Top Navigation Bar --- */
    header[data-testid="stHeader"] {
        background-color: #002744;
        color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    header[data-testid="stHeader"] * {
        color: white !important;
    }

    /* --- Sidebar Styling --- */
    section[data-testid="stSidebar"] {
        background-color: #d0f0f6;
        border-right: 2px solid #b2dfdb;
    }

    /* --- Headings --- */
    h1, h2, h3, h4 {
        color: #003366;
        font-weight: 600;
    }

    /* --- Metric Boxes --- */
    .stMetric {
        background-color: #d4f1f9 !important;
        padding: 18px !important;
        border-radius: 12px !important;
        border: 1px solid #b2ebf2 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.07) !important;
        text-align: center;
    }

    /* --- Misc Text Color Fixes --- */
    .css-1v0mbdj, .css-1r6slb0, .css-10trblm {
        color: #0a2e45;
    }

    /* --- Pie & Bar chart label fixes if needed --- */
    .legend text {
        fill: #003366;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🎨 Consistent Blue-Green Color Palette
colors = {
    "primary": "#00395d",  # Dark Blue
    "secondary": "#0288d1",  # Medium Blue
    "accent": "#81c784",  # Soft Green
    "profit": "#2f4b7c",  # Blue for profit-related items
    "expense": "#d32f2f",  # Red for expense-related items
    "neutral": "#f1f9f9",  # Light greyish-blue background
    "highlight": "#7c4dff",  # Bright purple for highlights
}

# -------------------------------
# 📥 Load data
sales = pd.read_csv("product-sales.csv", parse_dates=["Date"])
expenses = pd.read_csv("expenses.csv", parse_dates=["Date"])

sales["Amount"] = pd.to_numeric(sales["Amount"].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce")
expenses["Amount"] = pd.to_numeric(expenses["Amount"].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce")

sales["Date"] = pd.to_datetime(sales["Date"], format="%d/%m/%Y", errors="coerce")
expenses["Date"] = pd.to_datetime(expenses["Date"], format="%d/%m/%Y", errors="coerce")

sales = sales.dropna(subset=["Amount"])
expenses = expenses.dropna(subset=["Amount"])

# -------------------------------
# 🎛️ Sidebar Filters
st.sidebar.header("🔎 Filters")

min_date = min(sales["Date"].min(), expenses["Date"].min())
max_date = max(sales["Date"].max(), expenses["Date"].max())

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
product_filter = st.sidebar.multiselect("Select Product(s)", options=sales["Product"].unique(), default=sales["Product"].unique())
party_filter = st.sidebar.multiselect("Select Party", options=sales["Party Name"].unique(), default=sales["Party Name"].unique())
expense_type_filter = st.sidebar.multiselect("Select Expense Type(s)", options=expenses["Expense Type"].unique(), default=expenses["Expense Type"].unique())

# 🧼 Apply Filters
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

sales_filtered = sales[
    (sales["Date"].between(start_date, end_date)) &
    (sales["Product"].isin(product_filter)) &
    (sales["Party Name"].isin(party_filter))
]

expenses_filtered = expenses[
    (expenses["Date"].between(start_date, end_date)) &
    (expenses["Expense Type"].isin(expense_type_filter))
]

# -------------------------------
# 🎯 KPIs and Metrics
def format_inr(value):
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"

st.title("🧾 Kani Traders Business Dashboard")
st.subheader("📊 Business Metrics")
total_sales = sales_filtered['Amount'].sum()
total_expenses = expenses_filtered['Amount'].sum()
net_profit = total_sales - total_expenses
num_sales = len(sales_filtered)
num_expenses = len(expenses_filtered)
top_product = sales_filtered.groupby("Product")["Amount"].sum().idxmax()

k1, k2, k3 = st.columns(3)
k1.metric("Total Sales", format_inr(total_sales))
k2.metric("Total Expenses", format_inr(total_expenses))
k3.metric("Net Profit", format_inr(net_profit))

k4, k5 = st.columns(2)
k4.metric("Transactions (Sales)", num_sales)
k5.metric("Top Product", top_product)

# --- Average Sale Value & Profit Margin ---

col7, col8 = st.columns(2)
with col7:
    net_profit = total_sales - expenses_filtered['Amount'].sum()
    profit_margin = (net_profit / total_sales) * 100 if total_sales > 0 else 0
    st.metric(label="💰 Profit Margin", value=f"{profit_margin:.2f}%", delta=f"₹{net_profit:,.0f}")

with col8:
    sales_per_party = sales_filtered.groupby("Party Name")["Amount"].sum().reset_index()
    avg_sale_per_customer = sales_per_party["Amount"].sum() / sales_per_party.shape[0]
    st.metric("💸 Average Sale Value per Customer", format_inr(avg_sale_per_customer))


# -------------------------------
# 📊 Data Preparation for Plots
monthly_sales = sales_filtered.groupby(sales_filtered["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
monthly_sales['Date'] = monthly_sales['Date'].astype(str)
monthly_expenses = expenses_filtered.groupby(expenses_filtered["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
monthly_expenses['Date'] = monthly_expenses['Date'].astype(str)
merged = pd.merge(monthly_sales, monthly_expenses, on="Date", how="outer", suffixes=("_Sales", "_Expenses")).fillna(0)
merged['Amount_Sales_Lakh'] = merged['Amount_Sales'] / 1e5
merged['Amount_Expenses_Lakh'] = merged['Amount_Expenses'] / 1e5

st.subheader("📊 Sales Forecast")

# Load saved model and scaler
@st.cache_resource
def load_prediction_assets():
    model = load_model("rnn_sales_model.h5")
    scaler = joblib.load("sales_scaler.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata["sequence_length"]

model, scaler, SEQ_LENGTH = load_prediction_assets()

# Prepare monthly sales data
sales["Date"] = pd.to_datetime(sales["Date"])
monthly_sales = sales.set_index("Date").resample("M")["Amount"].sum().dropna()
product_data = monthly_sales.reset_index()

# Prepare initial sequence
recent_sales = product_data["Amount"].values[-SEQ_LENGTH:]

if len(recent_sales) < SEQ_LENGTH:
    st.warning(f"Not enough data for prediction (requires at least {SEQ_LENGTH} data points).")
else:
    # Scale and reshape input
    input_seq = scaler.transform(recent_sales.reshape(-1, 1)).flatten().tolist()
    future_preds = []

    for step in range(4):
        current_input = current_input 
        current_input = np.array(input_seq[-SEQ_LENGTH:]).reshape(1, SEQ_LENGTH, 1)
        pred_scaled = model.predict(current_input)
        pred_value = scaler.inverse_transform(pred_scaled)[0][0]
        future_preds.append(pred_value)

        # Append to input for next prediction
        input_seq.append(pred_scaled[0][0])  # Append scaled value for the next round

# Show the first prediction (formatted in Lakhs and Crores)
def format_inr_lakh_crore(value):
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"

# Create a DataFrame for plotting
plot_df = product_data.copy()
plot_df["Amount_Lakh"] = plot_df["Amount"] / 1e5  # Convert actual sales to Lakhs
plot_df["Type"] = "Actual"

# Future prediction dates
last_date = plot_df["Date"].iloc[-1]
future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(4)]

# Convert the predicted sales values to Lakhs for display on the plot
future_preds_lakh = [pred / 1e5 for pred in future_preds]

future_df = pd.DataFrame({
    "Date": future_dates,
    "Amount_Lakh": future_preds_lakh,  # Use the scaled values in Lakhs
    "Type": ["Predicted"] * 4
})

full_plot_df = pd.concat([plot_df, future_df], ignore_index=True)

# Plot the sales prediction graph with scaled values in Lakhs
fig_pred = px.line(
    full_plot_df,
    x="Date",
    y="Amount_Lakh",  # Plot the sales in Lakhs
    color="Type",
    color_discrete_map={"Actual": colors["primary"], "Predicted": colors["highlight"]},
    markers=True,
    title="Sales Forecast"
)

# Show the plot with the updated scale in Lakhs
st.plotly_chart(fig_pred, use_container_width=True)

# -------------------------------
# 📈 Charts Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Sales vs Expenses Over Time")
    fig1 = px.line(merged, x="Date", y=["Amount_Sales_Lakh", "Amount_Expenses_Lakh"],
                   markers=True, color_discrete_sequence=[colors["primary"], colors["expense"]])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📈 Monthly Profit Trend")
    merged["Profit_Lakh"] = merged["Amount_Sales_Lakh"] - merged["Amount_Expenses_Lakh"]
    fig4 = px.line(merged, x="Date", y="Profit_Lakh",
                   color_discrete_sequence=[colors["profit"]],
                   title="Profit Over Time  ")
    st.plotly_chart(fig4, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("🛍️ Product Sales Distribution")
    product_sales = sales_filtered.groupby("Product")["Amount"].sum().sort_values(ascending=False).reset_index()
    product_sales["Amount_Lakh"] = product_sales["Amount"] / 1e5
    fig9 = px.pie(product_sales, names="Product", values="Amount_Lakh",
                  color_discrete_sequence=[colors["primary"], colors["secondary"], colors["accent"], "#81c784", "#388e3c"])
    st.plotly_chart(fig9, use_container_width=True)

with col4:
    st.subheader("💸 Expense Breakdown")
    expense_type = expenses_filtered.groupby("Expense Type")["Amount"].sum().reset_index()
    expense_type["Amount_Lakh"] = expense_type["Amount"] / 1e5
    fig5 = px.pie(expense_type, names="Expense Type", values="Amount_Lakh", hole=0.5,
                  color_discrete_sequence=[colors["secondary"], colors["accent"], colors["primary"], colors["expense"], colors["highlight"]])
    fig5.update_traces(textinfo='percent+label')
    st.plotly_chart(fig5, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.subheader("📊 Products by Sales Volume")
    top_products_by_sales = product_sales.head(10)
    fig7 = px.bar(top_products_by_sales, x="Product", y="Amount_Lakh", title="Top 10 Products by Sales",
                  color="Amount_Lakh", color_continuous_scale=[colors["secondary"], colors["primary"]])
    st.plotly_chart(fig7, use_container_width=True)

with col6:
    st.subheader("🤝 Party-wise Sales")
    party_sales = sales_filtered.groupby("Party Name")["Amount"].sum().sort_values(ascending=False).reset_index()
    party_sales["Amount_Lakh"] = party_sales["Amount"] / 1e5
    fig3 = px.bar(party_sales.head(10), x="Party Name", y="Amount_Lakh", color="Amount_Lakh",
                  color_continuous_scale=[colors["accent"], colors["primary"]])
    st.plotly_chart(fig3, use_container_width=True)


# --- Cumulative Profit Over Time ---
st.subheader("📈 Cumulative Profit Over Time")
sales_daily = sales_filtered.groupby("Date")["Amount"].sum().cumsum().reset_index(name="Cumulative Sales")
expenses_daily = expenses_filtered.groupby("Date")["Amount"].sum().cumsum().reset_index(name="Cumulative Expenses")
cumulative = pd.merge(sales_daily, expenses_daily, on="Date", how="outer").sort_values("Date").fillna(method='ffill').fillna(0)
cumulative["Profit_Lakh"] = (cumulative["Cumulative Sales"] - cumulative["Cumulative Expenses"]) / 1e5
fig8 = px.line(cumulative, x="Date", y="Profit_Lakh",
               title="Cumulative Profit  ",
               color_discrete_sequence=[colors["profit"]])
st.plotly_chart(fig8, use_container_width=True)
