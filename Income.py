import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("WIID_Final_Cleaned_2010_2024.csv")
    df.columns = df.columns.str.strip()  # remove leading/trailing spaces
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
country_col = "country" if "country" in df.columns else df.columns[0]  # fallback to first column
country_list = ["All"] + sorted(df[country_col].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", country_list)

# REGION FILTER
# Pick region column if exists
region_options = [col for col in ["Region_WB", "Region_UN"] if col in df.columns]
if region_options:
    region_col = st.sidebar.selectbox(
        "Select Region Column used in dataset",
        region_options
    )
    region_list = ["All"] + sorted(df[region_col].dropna().unique())
    selected_region = st.sidebar.selectbox("Select Region", region_list)
else:
    region_col = None
    selected_region = "All"

# YEARS FILTER
if "year" in df.columns:
    years = sorted(df["year"].dropna().unique())
    year_range = st.sidebar.slider(
        "Select Year Range",
        int(min(years)), int(max(years)),
        (int(min(years)), int(max(years)))
    )
else:
    year_range = (2010, 2024)

# APPLY FILTERS
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df[country_col] == selected_country]

if region_col and selected_region != "All":
    filtered_df = filtered_df[filtered_df[region_col] == selected_region]

if "year" in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df["year"] >= year_range[0]) & (filtered_df["year"] <= year_range[1])]

# ==========================================
# METRICS / KPI CARDS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📍 Countries Analyzed", len(filtered_df[country_col].unique()))
col2.metric(
    "📊 Avg Gini Index",
    round(filtered_df["gini"].mean(), 2) if "gini" in filtered_df.columns else "N/A"
)
col3.metric(
    "💰 Avg GDP Per Capita",
    round(filtered_df["gdp_per_capita"].mean(), 2) if "gdp_per_capita" in filtered_df.columns else "N/A"
)
col4.metric(
    "📈 Avg Palma Ratio",
    round(filtered_df["palma_ratio"].mean(), 2) if "palma_ratio" in filtered_df.columns else "N/A"
)

# ==========================================
# VISUALIZATIONS
# ==========================================

# Gini Index Trend Over Years
if "gini" in filtered_df.columns and "year" in filtered_df.columns:
    st.subheader("📉 Gini Index Trend Over Years")
    line_fig = px.line(
        filtered_df,
        x="year",
        y="gini",
        color=country_col,
        markers=True,
        title="How Gini Index Changed Over Time"
    )
    st.plotly_chart(line_fig, use_container_width=True)

# Top 10 High Inequality Countries
if "gini" in df.columns:
    st.subheader("🏆 Countries with Highest Inequality (Top 10 by Gini)")
    top10 = df.groupby(country_col)["gini"].mean().sort_values(ascending=False).head(10).reset_index()
    bar_fig = px.bar(top10, x="gini", y=country_col, orientation="h", title="Top 10 High Inequality Countries")
    st.plotly_chart(bar_fig, use_container_width=True)

# Income Inequality vs GDP
if "gdp_per_capita" in filtered_df.columns and "gini" in filtered_df.columns:
    st.subheader("📌 Income Inequality vs GDP Per Capita")
    scatter_fig = px.scatter(
        filtered_df,
        x="gdp_per_capita",
        y="gini",
        size="population" if "population" in filtered_df.columns else None,
        color=region_col if region_col else None,
        hover_name=country_col,
        title="Relationship Between GDP & Inequality",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
