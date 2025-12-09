import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("WIID_Final_Cleaned_2010_2024.csv")
    return df

df = load_data()

# Show dataset columns for verification
st.write("### Columns in Dataset:")
st.write(df.columns.tolist())

# Main Title
st.title("🌍 Global Income Inequality Analysis Dashboard (2010-2024)")
st.write("Analyze global inequality using the WIID dataset with interactive filtering and visualizations.")

# Sidebar Filters
st.sidebar.header("Filters")

# Country Filter (assuming column name = 'Country')
if "Country" in df.columns:
    countries = st.sidebar.multiselect("Select Countries", df["Country"].unique())
else:
    st.error("Column 'Country' not found — please check dataset column names above.")
    st.stop()

# Year Range Filter
if "Year" in df.columns:
    min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
    years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
else:
    st.error("Column 'Year' not found — please check dataset column names above.")
    st.stop()

# Filter dataset
filtered_df = df[
    (df["Country"].isin(countries) if countries else True) &
    (df["Year"] >= years[0]) & (df["Year"] <= years[1])
]

# Display filtered data
st.write("### Filtered Data Preview:")
st.dataframe(filtered_df)

# ================== CHARTS ==================

# Gini Line Chart (if exists)
if "Gini" in df.columns and countries:
    st.write("### Gini Coefficient Trend Over Time")
    fig = px.line(filtered_df, x="Year", y="Gini", color="Country", markers=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select countries and ensure 'Gini' column exists to display trend chart.")

# Bar Chart: Average Gini Comparison
if "Gini" in df.columns and countries:
    st.write("### Average Gini Comparison for Selected Countries")
    avg_gini = filtered_df.groupby("Country")["Gini"].mean().reset_index()
    fig2 = px.bar(avg_gini, x="Country", y="Gini", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

# Summary Statistics
st.write("### Summary Statistics")
st.write(filtered_df.describe())

# Download button
st.write("### Download Filtered Dataset")
st.download_button(
    label="Download as CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="Filtered_Inequality_Data.csv",
    mime="text/csv"
)

st.success("Dashboard Loaded Successfully 🚀")
