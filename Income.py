import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Global Income Inequality Dashboard", layout="wide")

st.title("🌍 Global Income Inequality Analysis Dashboard (2010-2024)")
st.write("Upload the WIID dataset and analyze global inequality trends using interactive charts.")

# ================= FILE UPLOADER ====================

uploaded_file = st.file_uploader("📂 Upload WIID Dataset CSV File", type=["csv"])

if uploaded_file is not None:

    # Try different separators
    try:
        df = pd.read_csv(uploaded_file)
    except:
        try:
            df = pd.read_csv(uploaded_file, sep=";")
        except:
            df = pd.read_csv(uploaded_file, sep="\t")

    st.success("Dataset uploaded successfully! 🎉")

    # Show column names
    st.write("### 🧾 Columns in Dataset:")
    st.write(df.columns.tolist())

    # Preview data
    st.write("### 🔍 Dataset Preview:")
    st.dataframe(df.head())

    # ================= FILTERS ========================

    st.sidebar.header("🔎 FILTER OPTIONS")

    # Detect country column automatically
    possible_country_cols = ["Country", "country", "Nation", "Area", "region", "Region", "Country_Name"]
    country_col = None

    for col in possible_country_cols:
        if col in df.columns:
            country_col = col
            break

    if country_col is None:
        st.error("❌ No valid Country column found. Please check dataset.")
        st.stop()

    # Country filter
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        df[country_col].unique()
    )

    # Detect year column automatically
    if "Year" in df.columns:
        min_year = int(df["Year"].min())
        max_year = int(df["Year"].max())
        year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    else:
        st.error("❌ 'Year' column missing in dataset.")
        st.stop()

    # Filter dataset
    filtered_df = df[
        ((df[country_col].isin(selected_countries)) if selected_countries else True)
        & (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])
    ]

    st.write("### 📊 Filtered Data")
    st.dataframe(filtered_df)

    # ================= CHARTS ========================

    # Gini trend line chart
    if "Gini" in df.columns:
        st.write("### 📈 Gini Coefficient Trend Over Time")
        fig1 = px.line(filtered_df, x="Year", y="Gini", color=country_col, markers=True)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("⚠ 'Gini' column not found. Cannot show inequality trend chart.")

    # Average Gini Bar chart
    if "Gini" in df.columns:
        st.write("### 📊 Average Gini by Country")
        avg_gini = filtered_df.groupby(country_col)["Gini"].mean().reset_index()
        fig2 = px.bar(avg_gini, x=country_col, y="Gini", text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Summary statistics
    st.write("### 📑 Statistical Summary")
    st.write(filtered_df.describe())

    # Download filtered file
    st.write("### ⬇ Download Filtered Dataset")
    st.download_button(
        label="Download CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="Filtered_Inequality_Data.csv",
        mime="text/csv"
    )

    st.success("Dashboard Ready 🎉")
else:
    st.info("👆 Please upload the dataset to begin analysis.")
