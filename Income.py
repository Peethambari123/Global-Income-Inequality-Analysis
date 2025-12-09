import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Global Inequality Dashboard",
                   page_icon="🌍",
                   layout="wide")

# Load dataset
df = pd.read_csv("WIID_Final_Cleaned_2010_2024.csv")

# Display Columns for debugging
st.sidebar.write("📌 Dataset Columns:")
st.sidebar.write(list(df.columns))

# Rename columns based on dataset (change if needed)
df.rename(columns={
    "Country": "country",
    "Year": "year",
    "Gini": "gini",
    "GDP_Per_Capita": "gdp",
    "Region": "region",
    "Palma_Ratio": "palma",
    "Income_Group": "income_group"
}, inplace=True)

# Sidebar
st.sidebar.title("🌍 Global Inequality Dashboard")

region_filter = st.sidebar.selectbox("Select Region", options=["All"] + list(df["region"].unique()))
country_filter = st.sidebar.multiselect("Select Countries", options=df["country"].unique())
income_filter = st.sidebar.multiselect("Select Income Groups", options=df["income_group"].unique())

# Apply Filters
data = df.copy()
if region_filter != "All":
    data = data[data["region"] == region_filter]
if len(country_filter) > 0:
    data = data[data["country"].isin(country_filter)]
if len(income_filter) > 0:
    data = data[data["income_group"].isin(income_filter)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Countries", len(data["country"].unique()))
col2.metric("Avg Gini Index", f"{data['gini'].mean():.2f}")
col3.metric("Avg GDP per Capita", f"{data['gdp'].mean():,.2f} $")
col4.metric("Avg Palma Ratio", f"{data['palma'].mean():.2f}")

st.markdown("### 🌎 Global Inequality Analysis")

# Charts
colA, colB = st.columns(2)

with colA:
    st.subheader("World Map - Gini Index")
    fig_map = px.choropleth(data,
                            locations="country",
                            locationmode="country names",
                            color="gini",
                            hover_name="country",
                            title="Gini Index by Country",
                            color_continuous_scale="Viridis")
    st.plotly_chart(fig_map, use_container_width=True)

with colB:
    st.subheader("Countries by Income Class")
    fig_pie = px.pie(data, names="income_group", title="Country Income Group Share")
    st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("📈 Trends Over Time")
fig_trend = px.line(data, x="year", y="gini", color="country", title="Gini Trend Over Years")
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("Relationship Between GDP & Inequality")
fig_scatter = px.scatter(data, x="gdp", y="gini", color="income_group",
                         size="palma", title="GDP vs Gini Index")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Top Countries by Highest Inequality")
top = data.groupby("country")["gini"].mean().sort_values(ascending=False).head(10).reset_index()
fig_bar = px.bar(top, x="country", y="gini", title="Highest Gini Countries")
st.plotly_chart(fig_bar, use_container_width=True)
