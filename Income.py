# Income.py - Global Income Inequality Streamlit App
# Requirements: streamlit, pandas, plotly, numpy, pycountry
# Run: streamlit run Income.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pycountry
from io import StringIO

st.set_page_config(page_title="Global Inequality Dashboard", layout="wide")

# --------------------- Utility functions ---------------------

def robust_read_csv(uploaded, local_path="WIID_Final_Cleaned_2010_2024.csv"):
    """
    Try multiple ways to read the CSV: uploaded file object first, then local path.
    Return a DataFrame or raise an informative Exception.
    """
    errs = []
    df = None
    # If user uploaded via Streamlit uploader
    if uploaded is not None:
        # uploaded is BytesIO / UploadedFile
        for sep in [",", ";", "\t"]:
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, sep=sep, engine='python', encoding='utf-8')
                # if only one column and contains text indicating code, try fallback encoding
                if df.shape[1] == 1 and df.columns[0].strip().startswith("df = pd.read_csv"):
                    # this looks like wrong file content (script). continue trying.
                    errs.append("Uploaded file appears to contain script/text instead of CSV data.")
                    df = None
                    continue
                return df
            except Exception as e:
                errs.append(f"sep={sep} -> {e}")
        # try latin1 encoding
        for sep in [",", ";", "\t"]:
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, sep=sep, engine='python', encoding='latin1')
                if df.shape[1] == 1 and df.columns[0].strip().startswith("df = pd.read_csv"):
                    errs.append("Uploaded file appears to contain script/text instead of CSV data.")
                    df = None
                    continue
                return df
            except Exception as e:
                errs.append(f"latin1 sep={sep} -> {e}")
    # Try local file path fallback
    try:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(local_path, sep=sep, engine='python', encoding='utf-8')
                return df
            except Exception:
                pass
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(local_path, sep=sep, engine='python', encoding='latin1')
                return df
            except Exception:
                pass
    except FileNotFoundError:
        errs.append(f"Local file {local_path} not found.")
    raise Exception("Unable to read CSV. Tried multiple separators/encodings. Errors:\n" + "\n".join(errs))


def infer_columns(df):
    """
    Identify best matching column names for country, year, gini, gdp, palma, region, income_group.
    Returns a dict mapping canonical names to actual df column names (or None)
    """
    cols = [c for c in df.columns]
    lower = {c.lower(): c for c in cols}
    mapping = {}
    # Candidate sets
    candidate = {
        "country": ["country", "country_name", "countries", "nation", "countryname", "location", "countrycode"],
        "year": ["year", "yr", "time"],
        "gini": ["gini", "gini_index", "gini index", "gini_coeff"],
        "gdp": ["gdp_per_capita", "gdp per capita", "gdp_pc", "gdp_pc_usd", "gdp"],
        "palma": ["palma", "palma_ratio", "palma ratio"],
        "region": ["region", "world_bank_region", "region_wb", "world region"],
        "income_group": ["income_group", "income group", "income_class", "income"]
    }
    for key, names in candidate.items():
        found = None
        for n in names:
            if n in lower:
                found = lower[n]
                break
        # Try partial matches
        if found is None:
            for col in cols:
                if col.lower().replace(" ", "_") in [x.replace(" ", "_") for x in names]:
                    found = col
                    break
        mapping[key] = found
    return mapping


def safe_cast_year(df, year_col):
    try:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
    except Exception:
        try:
            df[year_col] = pd.to_datetime(df[year_col], errors='coerce').dt.year
            df[year_col] = df[year_col].astype('Int64')
        except Exception:
            pass
    return df


def clean_country_names_for_choropleth(df, country_col):
    """
    Try to map country names to pycountry standard names; create a 'country_for_map' column.
    """
    def try_resolve(name):
        if pd.isna(name): return None
        name = str(name).strip()
        # Quick manual fixes common mismatches
        replacements = {
            "United States of America": "United States",
            "USA": "United States",
            "UK": "United Kingdom",
            "Korea, Rep.": "Korea, Republic of",
            "Republic of Korea": "Korea, Republic of",
            "Viet Nam": "Vietnam",
            "Russian Federation": "Russia",
            "Czechia": "Czech Republic"
        }
        if name in replacements:
            return replacements[name]
        # Try pycountry lookup
        try:
            r = pycountry.countries.lookup(name)
            return r.name
        except Exception:
            # fallback return original name
            return name

    df['country_for_map'] = df[country_col].apply(try_resolve)
    return df


def simple_ai_summary(df, mapping):
    """
    Construct a short rule-based summary about inequality trends from filtered df.
    """
    parts = []
    year_col = mapping.get("year")
    gini_col = mapping.get("gini")
    country_col = mapping.get("country")

    if gini_col is None or year_col is None:
        return "AI Insights require 'Gini' and 'Year' columns in dataset."

    overall = df.copy()
    if overall.empty:
        return "No data selected to analyze."

    # compute recent years trend
    recent = overall[overall[year_col] >= overall[year_col].max() - 4]  # last 5 years
    mean_recent = recent[gini_col].mean() if not recent.empty else np.nan
    mean_all = overall[gini_col].mean()

    parts.append(f"Across the selected data, the average Gini index is {mean_all:.2f} (recent 5-years avg: {mean_recent:.2f}).")

    # top and bottom countries
    top = overall.groupby(country_col)[gini_col].mean().sort_values(ascending=False).head(3)
    bottom = overall.groupby(country_col)[gini_col].mean().sort_values(ascending=True).head(3)
    parts.append("Highest average inequality (top 3): " + ", ".join([f"{c} ({top[c]:.2f})" for c in top.index]))
    parts.append("Lowest average inequality (bottom 3): " + ", ".join([f"{c} ({bottom[c]:.2f})" for c in bottom.index]))

    # trend direction
    try:
        # take median gini by year
        trend = overall.groupby(year_col)[gini_col].median().sort_index()
        if len(trend) >= 2:
            if trend.iloc[-1] > trend.iloc[0]:
                parts.append("Median Gini has increased over the period selected.")
            else:
                parts.append("Median Gini has decreased or remained stable over the period selected.")
    except Exception:
        pass

    return " ".join(parts)

# --------------------- App UI ---------------------

st.title("🌍 Global Income Inequality Dashboard (2010-2024)")
st.write("Upload the WIID CSV (or place `WIID_Final_Cleaned_2010_2024.csv` in app folder). This app tries to auto-detect columns and separators.")

# File uploader + local fallback
uploaded_file = st.file_uploader("Upload WIID CSV file", type=["csv"])
df = None
try:
    df = robust_read_csv(uploaded_file)
except Exception as e:
    st.error("Failed to read dataset.")
    st.info("Make sure you uploaded the real CSV (not a .py or text file). If you used upload, try re-uploading. If you keep the file locally, place it in the app folder with filename:\n`WIID_Final_Cleaned_2010_2024.csv`")
    st.text(str(e))
    st.stop()

# Quick check: if df seems like a single column containing code text -> stop and show guidance
if df.shape[1] == 1:
    only_col = df.columns[0]
    # If the cell contains 'df = pd.read_csv' it's very likely wrong file
    if any("df = pd.read_csv" in str(v) for v in df.iloc[:,0].astype(str).head(3)):
        st.error("It looks like you uploaded the wrong file (a script or text file). Please upload the actual CSV dataset.")
        st.stop()

# Show detected columns for user verification
st.sidebar.header("Dataset Info")
st.sidebar.write("Columns detected:")
st.sidebar.write(list(df.columns))

# Infer columns
mapping = infer_columns(df)
st.sidebar.write("Auto-detected column mapping (may be 'None'):")
st.sidebar.write(mapping)

# Validate minimum required columns
if mapping.get("country") is None or mapping.get("year") is None:
    st.error("Couldn't automatically detect essential columns (country / year). Please ensure your CSV has country and year columns.")
    st.stop()

# Normalize year column to numeric
df = safe_cast_year(df, mapping["year"])

# Rename the columns internally to canonical names for ease
canonical = {}
for key, col in mapping.items():
    if col is not None:
        canonical[key] = col

# Create a copy with canonical names (if present)
work = df.copy()
if mapping.get("country"):
    work.rename(columns={mapping["country"]: "country"}, inplace=True)
if mapping.get("year"):
    work.rename(columns={mapping["year"]: "year"}, inplace=True)
if mapping.get("gini"):
    work.rename(columns={mapping["gini"]: "Gini"}, inplace=True)
if mapping.get("gdp"):
    work.rename(columns={mapping["gdp"]: "gdp"}, inplace=True)
if mapping.get("palma"):
    work.rename(columns={mapping["palma"]: "palma"}, inplace=True)
if mapping.get("region"):
    work.rename(columns={mapping["region"]: "region"}, inplace=True)
if mapping.get("income_group"):
    work.rename(columns={mapping["income_group"]: "income_group"}, inplace=True)

# Create minimal checks for numeric columns
if "Gini" in work.columns:
    work["Gini"] = pd.to_numeric(work["Gini"], errors='coerce')
if "gdp" in work.columns:
    work["gdp"] = pd.to_numeric(work["gdp"], errors='coerce')
if "palma" in work.columns:
    work["palma"] = pd.to_numeric(work["palma"], errors='coerce')

# Clean country names for mapping
work = clean_country_names_for_choropleth(work, "country")

# Sidebar navigation (pages)
page = st.sidebar.radio("Navigate", ["Home", "Dashboard", "Country Compare", "Trends", "AI Insights", "About"])

# Shared filters (region, income group, countries, year range)
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
regions = ["All"] + sorted(work["region"].dropna().unique().tolist()) if "region" in work.columns else ["All"]
region_sel = st.sidebar.selectbox("Region", regions)

income_options = sorted(work["income_group"].dropna().unique().tolist()) if "income_group" in work.columns else []
income_sel = st.sidebar.multiselect("Income Group", options=income_options, default=income_options)

countries_list = sorted(work["country"].dropna().unique().tolist())
country_sel = st.sidebar.multiselect("Countries (optional)", options=countries_list, default=[])

# Year slider
min_year = int(work["year"].min()) if work["year"].notna().any() else 2010
max_year = int(work["year"].max()) if work["year"].notna().any() else 2024
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

# Apply filters to produce 'filtered' df
filtered = work.copy()
if region_sel != "All" and "region" in filtered.columns:
    filtered = filtered[filtered["region"] == region_sel]
if income_options:
    if income_sel:
        filtered = filtered[filtered["income_group"].isin(income_sel)]
if country_sel:
    filtered = filtered[filtered["country"].isin(country_sel)]
filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

# --------------------- Pages ---------------------

if page == "Home":
    st.header("Welcome")
    st.write("This dashboard helps explore global income inequality (WIID dataset). Use the left sidebar to upload/select filters and navigate pages.")
    st.write("If the dataset columns shown in the sidebar don't match expected names (country, year, gini), re-upload a correct CSV or rename columns in the CSV accordingly.")
    st.write("Quick tips:")
    st.markdown("""
    - Upload CSV via the uploader at top if you haven't.
    - Ensure dataset has columns for **country** and **year** and ideally **Gini**, **gdp**, **palma**, **region**, **income_group**.
    - Use the *Dashboard* for KPIs and overview; *Country Compare* to compare specific countries; *Trends* for time-series; *AI Insights* for auto summaries.
    """)

elif page == "Dashboard":
    st.header("Dashboard Overview")

    # KPIs row
    col1, col2, col3, col4 = st.columns(4)
    total_countries = filtered["country"].nunique()
    avg_gini = filtered["Gini"].mean() if "Gini" in filtered.columns else np.nan
    avg_gdp = filtered["gdp"].mean() if "gdp" in filtered.columns else np.nan
    avg_palma = filtered["palma"].mean() if "palma" in filtered.columns else np.nan

    col1.metric("Total Countries Analyzed", int(total_countries))
    col2.metric("Average Gini Index", f"{avg_gini:.2f}" if not np.isnan(avg_gini) else "N/A")
    col3.metric("Average GDP per Capita (USD)", f"{avg_gdp:,.0f}" if not np.isnan(avg_gdp) else "N/A")
    col4.metric("Average Palma Ratio", f"{avg_palma:.2f}" if not np.isnan(avg_palma) else "N/A")

    st.markdown("---")

    # Top row: choropleth and pie
    left, right = st.columns([2,1])
    with left:
        st.subheader("World Map: Gini Index")
        if "Gini" in filtered.columns:
            map_df = filtered.groupby("country_for_map", dropna=True)["Gini"].mean().reset_index()
            fig_map = px.choropleth(map_df, locations="country_for_map",
                                    locationmode="country names",
                                    color="Gini",
                                    hover_name="country_for_map",
                                    color_continuous_scale="Inferno",
                                    title="Average Gini by Country (selected filters)")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Gini column not found. World map requires 'Gini'.")

    with right:
        st.subheader("Countries by Income Group")
        if "income_group" in filtered.columns:
            pie = filtered.groupby("income_group")["country"].nunique().reset_index()
            pie.columns = ["income_group", "count"]
            fig_pie = px.pie(pie, names="income_group", values="count", title="Country Count by Income Group")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No income_group column found.")
    
    st.markdown("---")

    # Trends & scatter
    st.subheader("Trends & Relationships")
    t1, t2 = st.columns(2)
    with t1:
        st.write("Gini Trend Over Time")
        if "Gini" in filtered.columns:
            trend_df = filtered.groupby(["year"], as_index=False)["Gini"].median()
            fig_tr = px.line(trend_df, x="year", y="Gini", title="Median Gini Over Time")
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info("Gini data missing for trend.")
    with t2:
        st.write("GDP per Capita vs Gini")
        if "gdp" in filtered.columns and "Gini" in filtered.columns:
            fig_sc = px.scatter(filtered, x="gdp", y="Gini", color="income_group" if "income_group" in filtered.columns else None,
                                size="palma" if "palma" in filtered.columns else None, hover_name="country",
                                title="GDP vs Gini (bubble size = Palma)")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("gdp and/or Gini column is missing for scatter plot.")

    st.markdown("---")
    st.subheader("Top 10 Countries by Average Gini")
    if "Gini" in filtered.columns:
        top10 = filtered.groupby("country")["Gini"].mean().sort_values(ascending=False).reset_index().head(10)
        fig_bar = px.bar(top10, x="country", y="Gini", title="Top 10 Highest Average Gini")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Gini column required for this chart.")

    st.markdown("---")
    st.write("Filtered data preview:")
    st.dataframe(filtered.head(200))

    st.download_button("Download Filtered CSV", filtered.to_csv(index=False).encode('utf-8'), file_name="filtered_wiid.csv", mime="text/csv")

elif page == "Country Compare":
    st.header("Country Comparison")
    st.write("Select 2-5 countries to compare key metrics side-by-side.")

    compare_countries = st.multiselect("Compare Countries", options=countries_list, default=(countries_list[:3] if len(countries_list)>=3 else countries_list))
    compare_year = st.selectbox("Comparison Year (choose one)", options=sorted(work['year'].dropna().unique().tolist(), reverse=True))
    if compare_countries and compare_year:
        comp_df = work[(work["country"].isin(compare_countries)) & (work["year"] == compare_year)]
        if comp_df.empty:
            st.info("No data for selected countries in that year.")
        else:
            cols = st.columns(len(compare_countries))
            for i, c in enumerate(compare_countries):
                row = comp_df[comp_df["country"] == c]
                with cols[i]:
                    st.subheader(c)
                    gini_val = row["Gini"].mean() if "Gini" in row.columns and not row["Gini"].isna().all() else None
                    gdp_val = row["gdp"].mean() if "gdp" in row.columns and not row["gdp"].isna().all() else None
                    palma_val = row["palma"].mean() if "palma" in row.columns and not row["palma"].isna().all() else None
                    st.metric("Gini", f"{gini_val:.2f}" if gini_val is not None and not np.isnan(gini_val) else "N/A")
                    st.metric("GDP per Capita", f"{gdp_val:,.0f}" if gdp_val is not None and not np.isnan(gdp_val) else "N/A")
                    st.metric("Palma Ratio", f"{palma_val:.2f}" if palma_val is not None and not np.isnan(palma_val) else "N/A")
            # Comparison charts
            st.subheader("Comparison Charts Across Selected Countries")
            if "Gini" in comp_df.columns:
                fig_comp = px.line(work[work["country"].isin(compare_countries)], x="year", y="Gini", color="country", title="Gini Over Time")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Gini column missing.")

elif page == "Trends":
    st.header("Trends")
    st.write("Deep dive time-series visualizations.")

    metric = st.selectbox("Metric to plot", options=[c for c in ["Gini", "gdp", "palma"] if c in work.columns], index=0 if "Gini" in work.columns else 0)
    if metric:
        st.subheader(f"{metric} trend by country")
        # allow selecting few countries to show lines
        show_countries = st.multiselect("Show countries (max 8)", options=countries_list, default=(countries_list[:6] if len(countries_list)>=6 else countries_list))
        trend_df = work.copy()
        if show_countries:
            trend_df = trend_df[trend_df["country"].isin(show_countries)]
        fig_trend = px.line(trend_df, x="year", y=metric, color="country", title=f"{metric} over time")
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap (numeric columns)")
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = work[numeric_cols].corr()
        fig_heat = px.imshow(corr, text_auto=True, title="Correlation matrix")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

elif page == "AI Insights":
    st.header("AI-Powered Insights (simple rule-based)")
    st.write("This section provides automated, explainable summaries based on the filtered data. It's a local rule-based summarizer — no external APIs used.")

    if filtered.empty:
        st.info("No data selected. Adjust filters to generate insights.")
    else:
        summary = simple_ai_summary(filtered, mapping)
        st.subheader("Summary")
        st.write(summary)
        st.markdown("---")
        st.subheader("Top Signals")
        # Show simple signals: biggest year-to-year change in median Gini
        if "Gini" in filtered.columns:
            year_med = filtered.groupby("year")["Gini"].median().sort_index()
            if len(year_med) >= 2:
                changes = year_med.diff().abs().sort_values(ascending=False)
                top_change_year = changes.index[0]
                st.write(f"Largest year-to-year median-Gini change occurred around year {int(top_change_year)} (change = {changes.iloc[0]:.2f})")
            # show top rising countries last 5 years
            last_year = filtered["year"].max()
            recent = filtered[filtered["year"] >= last_year-4]
            if not recent.empty:
                change_by_country = recent.groupby("country")["Gini"].agg(['first','last'])
                change_by_country = change_by_country.dropna()
                if not change_by_country.empty:
                    change_by_country['delta'] = change_by_country['last'] - change_by_country['first']
                    rising = change_by_country.sort_values('delta', ascending=False).head(5)
                    st.write("Countries with largest rise in Gini (recent period):")
                    st.table(rising['delta'].reset_index().rename(columns={'delta':'Gini change'}))
        else:
            st.info("Gini column needed for AI insights.")

elif page == "About":
    st.header("About this App")
    st.markdown("""
    **Global Income Inequality Dashboard**
    - Built with Streamlit and Plotly.
    - Upload your WIID CSV file or place the dataset file in the same folder named `WIID_Final_Cleaned_2010_2024.csv`.
    - The app auto-detects common column names; please ensure country and year columns exist.
    - Author: (You). Customize further as needed.
    """)
    st.markdown("**Notes & Troubleshooting**")
    st.markdown("""
    - If you see only a single column that contains text like `df = pd.read_csv(...)`, you likely uploaded a `.py` file or a text snippet. Re-upload the actual CSV.
    - For mapping problems, ensure country names are standard. The app attempts to normalize names with `pycountry`.
    - Add additional columns or rename columns within your CSV to match expected fields if auto-detection fails.
    """)

# --------------------- End ---------------------
