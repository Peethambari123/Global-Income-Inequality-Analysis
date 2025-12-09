import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("WIID_Final_Cleaned_2010_2024.csv")
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ==========================================
# AUTO-DETECT COLUMNS
# ==========================================
columns_lower = [c.lower() for c in df.columns]

# Country
country_col = None
for c in df.columns:
    if "country" in c.lower():
        country_col = c
        break
if not country_col:
    country_col = df.columns[0]  # fallback to first column

# Year
year_col = None
for c in df.columns:
    if "year" in c.lower():
        year_col = c
        break

# Gini
gini_col = None
for c in df.columns:
    if "gini" in c.lower():
        gini_col = c
        break

# GDP per Capita
gdp_col = None
for c in df.columns:
    if "gdp" in c.lower() and "capita" in c.lower():
        gdp_col = c
        break

# Palma Ratio
palma_col = None
for c in df.columns:
    if "palma" in c.lower():
        palma_col = c
        break

# Population
pop_col = None
for c in df.columns:
    if "population" in c.lower():
        pop_col = c
        break

# Region (WB or UN)
region_options = [c for c in df.columns if "region" in c.lower()]
region_col = region_options[0] if region_options else None

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
country_list = ["All"] + sorted(df[country_col].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", country_list)

# REGION FILTER
if region_col:
    region_list = ["All"] + sorted(df[region_col].dropna().unique())
    selected_region = st.sidebar.selectbox("Select Region", region_list)
else:
    selected_region = "All"

# YEARS FILTER
if year_col:
    years = sorted(df[year_col].dropna().unique())
    year_range = st.sidebar.slider(
        "Select Year Range",
        int(min(years)), int(max(years)),
        (int(min(years)), int(max(years)))
    )
else:
    year_range = (2010, 2024)

# ==========================================
# APPLY FILTERS
# ==========================================
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df[country_col] == selected_country]

if region_col and selected_region != "All":
    filtered_df = filtered_df[filtered_df[region_col] == selected_region]

if year_col:
    filtered_df = filtered_df[(filtered_df[year_col] >= year_range[0]) & (filtered_df[year_col] <= year_range[1])]

# ==========================================
# METRICS / KPI CARDS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📍 Countries Analyzed", len(filtered_df[country_col].unique()))
col2.metric(
    "📊 Avg Gini Index",
    round(filtered_df[gini_col].mean(), 2) if gini_col else "N/A"
)
col3.metric(
    "💰 Avg GDP Per Capita",
    round(filtered_df[gdp_col].mean(), 2) if gdp_col else "N/A"
)
col4.metric(
    "📈 Avg Palma Ratio",
    round(filtered_df[palma_col].mean(), 2) if palma_col else "N/A"
)

# ==========================================
# VISUALIZATIONS
# ==========================================
# Gini Index Trend Over Years
if gini_col and year_col:
    st.subheader("📉 Gini Index Trend Over Years")
    line_fig = px.line(
        filtered_df,
        x=year_col,
        y=gini_col,
        color=country_col,
        markers=True,
        title="How Gini Index Changed Over Time"
    )
    st.plotly_chart(line_fig, use_container_width=True)

# Top 10 High Inequality Countries
if gini_col:
    st.subheader("🏆 Countries with Highest Inequality (Top 10 by Gini)")
    top10 = df.groupby(country_col)[gini_col].mean().sort_values(ascending=False).head(10).reset_index()
    bar_fig = px.bar(top10, x=gini_col, y=country_col, orientation="h", title="Top 10 High Inequality Countries")
    st.plotly_chart(bar_fig, use_container_width=True)

# Income Inequality vs GDP
if gdp_col and gini_col:
    st.subheader("📌 Income Inequality vs GDP Per Capita")
    scatter_fig = px.scatter(
        filtered_df,
        x=gdp_col,
        y=gini_col,
        size=pop_col if pop_col else None,
        color=region_col if region_col else None,
        hover_name=country_col,
        title="Relationship Between GDP & Inequality",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
