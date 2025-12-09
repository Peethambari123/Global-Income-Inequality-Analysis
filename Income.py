import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('WIID_Final_Cleaned_2010_2024.csv')
    return df

df = load_data()

# Title
st.title("🌍 Global Inequality Analysis Dashboard (2010-2024)")
st.write("Analyze income inequality data across countries using WIID dataset.")

# Sidebar filters
st.sidebar.header("Filter Options")

countries = st.sidebar.multiselect(
    "Select Countries", df['country'].unique())

measure = st.sidebar.selectbox(
    "Select Inequality Measure", df['welfare_measure'].unique())

# Filter data
filtered_df = df.copy()
if countries:
    filtered_df = filtered_df[filtered_df['country'].isin(countries)]

if measure:
    filtered_df = filtered_df[filtered_df['welfare_measure'] == measure]

st.write("### Filtered Dataset Preview")
st.dataframe(filtered_df.head())

# Plotting
if not filtered_df.empty:
    st.write("### Inequality Trend Over Time")
    fig, ax = plt.subplots()
    for c in filtered_df['country'].unique():
        subset = filtered_df[filtered_df['country'] == c]
        ax.plot(subset['year'], subset['gini_reported'], label=c)
    ax.set_xlabel('Year')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Gini Coefficient Trend')
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("No data available for selected filters.")
