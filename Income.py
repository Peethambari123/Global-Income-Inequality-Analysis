import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("WIID_Final_Cleaned_2010_2024.csv")
    return df

df = load_data()

# ==========================================
# PAGE TITLE
# ==========================================
st.set_page_config(page_title="Global Inequality Dashboard", layout="wide")
st.title("🌍 Global Income Inequality Dashboard (2010-2024)")
st.write("Interactive visual analytics using WIID dataset")

# ==========================================
# SHOW COLUMNS FOR DEBUGGING
# ==========================================
with st.expander("📌 Dataset Column Names"):
    st.write(df.columns.tolist())

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Filters")

# COUNTRY FILTER
country_list = ["All"] + sorted(df["country"].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", country_list)

# REGION FILTER
region_col = st.sidebar.selectbox(
    "Select Region Column used in dataset",
    ["Region_WB", "Region_UN"],  # change names based on your file
)

region_list = ["All"] + sorted(df[region_col].dropna().unique())
selected_region = st.sidebar.selectbox("Select Region", region_list)

# YEARS FILTER
years = sorted(df["year"].dropna().unique())
year_range = st.sidebar.slider("Select Year Range", int(min(years)), int(max(years)), (2010, 2024))

# APPLY FILTERS
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]

if selected_region != "All":
    filtered_df = filtered_df[filtered_df[region_col] == selected_region]

filtered_df = filtered_df[(filtered_df["year"] >= year_range[0]) & (filtered_df["year"] <= year_range[1])]

# ==========================================
# METRICS / KPI CARDS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📍 Countries Analyzed", len(filtered_df["country"].unique()))
col2.metric("📊 Avg Gini Index", round(filtered_df["gini"].mean(), 2))
col3.metric("💰 Avg GDP Per Capita", round(filtered_df["gdp_per_capita"].mean(), 2))
col4.metric("📈 Avg Palma Ratio", round(filtered_df["palma_ratio"].mean(), 2))

# ==========================================
# VISUALIZATIONS
# ==========================================

st.subheader("📉 Gini Index Trend Over Years")
line_fig = px.line(
    filtered_df,
    x="year",
    y="gini",
    color="country",
    markers=True,
    title="How Gini Index Changed Over Time"
)
st.plotly_chart(line_fig, use_container_width=True)

# BAR CHART
st.subheader("🏆 Countries with Highest Inequality (Top 10 by Gini)")
top10 = df.groupby("country")["gini"].mean().sort_values(ascending=False).head(10).reset_index()
bar_fig = px.bar(top10, x="gini", y="country", orientation="h", title="Top 10 High Inequality Countries")
st.plotly_chart(bar_fig, use_container_width=True)

# SCATTER GDP vs GINI
st.subheader("📌 Income Inequality vs GDP Per Capita")
scatter_fig = px.scatter(
    filtered_df,
    x="gdp_per_capita",
    y="gini",
    size="population",
    color=region_col,
    hover_name="country",
    title="Relationship Between GDP & Inequality",
)
st.plotly_chart(scatter_fig, use_container_width=True)

