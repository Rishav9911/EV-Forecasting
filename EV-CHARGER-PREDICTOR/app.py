import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.tsa.api as tsa

# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="EV & Charger Demand – Washington State",
    layout="wide"
)

st.title("⚡ Leading the Charge: Predicting EV & Charger Demand in Washington State")

st.markdown(
    """
    This Streamlit app reproduces the analysis and modeling from your notebook:

    - Cleans and processes **EV title & registration** data  
    - Aggregates **EVs on the road by county and over time**  
    - Fits **SARIMAX models** per county using tuned parameters  
    - Loads **charging infrastructure counts by county**  
    - Compares **EV demand vs charger supply** by county  
    - Highlights **top counties to invest in chargers**
    """
)

# -------------------------------------------------------------------
# CONSTANTS & GLOBAL CONFIG
# -------------------------------------------------------------------

DATA_DIR = "data"
EV_FILE = os.path.join(DATA_DIR, "title_transactions-06-29-2021.csv.gz")
# >>> IMPORTANT: this is your pre-aggregated charger count file <<<
CHARGER_FILE = os.path.join(DATA_DIR, "df_charger_counts-07-13-2021.csv")

TOP_TEN_COUNTIES = ['King', 'Snohomish', 'Pierce', 'Clark', 'Thurston',
                    'Kitsap', 'Whatcom', 'Spokane', 'Benton', 'Island']

# Fixed SARIMAX parameters from the original notebook
MODEL_CONFIG = {
    "King":      dict(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12)),
    "Snohomish": dict(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)),
    "Pierce":    dict(order=(4, 1, 1), seasonal_order=(1, 0, 1, 12)),
    "Clark":     dict(order=(1, 1, 2), seasonal_order=(1, 0, 0, 12)),   # (1,0,[],12) → (1,0,0,12)
    "Thurston":  dict(order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)),
    "Kitsap":    dict(order=(4, 1, 0), seasonal_order=(0, 0, 0, 0)),   # no seasonality
    "Whatcom":   dict(order=(2, 1, 0), seasonal_order=(0, 0, 0, 0)),   # no seasonality
    "Spokane":   dict(order=(3, 1, 0), seasonal_order=(1, 0, 1, 12)),
    "Benton":    dict(order=(4, 1, 1), seasonal_order=(0, 0, 1, 12)),
    "Island":    dict(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),   # no seasonality
}

TRAIN_SPLIT = {
    "King": 0.80,
    "Snohomish": 0.75,
    "Pierce": 0.75,
    "Clark": 0.75,
    "Thurston": 0.80,
    "Kitsap": 0.75,
    "Whatcom": 0.75,
    "Spokane": 0.75,
    "Benton": 0.75,
    "Island": 0.70,
}

sns.set(style="whitegrid")

# -------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def load_ev_raw() -> pd.DataFrame:
    df = pd.read_csv(EV_FILE, compression="gzip", index_col=0)
    return df


def clean_ev_data(df: pd.DataFrame) -> pd.DataFrame:
    # transaction_date to datetime
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # m/y col
    df["m/y"] = df["transaction_date"].dt.strftime("%m-%Y")

    # set index to date
    df = df.set_index("transaction_date")

    # Drop government / tax-relevant columns
    drop_cols = [
        'electric_vehicle_fee_paid',
        'hb_2042_clean_alternative_fuel_vehicle_cafv_eligibility',
        'meets_2019_hb_2042_electric_range_requirement',
        'meets_2019_hb_2042_sale_date_requirement',
        'meets_2019_hb_2042_sale_price_value_requirement',
        'transportation_electrification_fee_paid',
        'hybrid_vehicle_electrification_fee_paid',
        'legislative_district',
        'non_clean_alternative_fuel'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Drop redundant columns
    drop_cols_2 = ['transaction_year', 'base_msrp', 'date_of_vehicle_sale']
    df = df.drop(columns=[c for c in drop_cols_2 if c in df.columns], errors="ignore")

    # Filter to title transactions (new + used purchases)
    df = df[(df["transaction_type"] == "Original Title") |
            (df["transaction_type"] == "Transfer Title")]

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove duplicates by (m/y, dol_vehicle_id, county)
    if {"m/y", "dol_vehicle_id", "county"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["m/y", "dol_vehicle_id", "county"], keep="last")

    # Remove duplicates by (m/y, dol_vehicle_id)
    if {"m/y", "dol_vehicle_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["m/y", "dol_vehicle_id"], keep="last")

    # Handle nulls: drop where county is NaN
    df = df.dropna(subset=["county"])

    # Filter to WA state
    if "state_of_residence" in df.columns:
        df = df[df["state_of_residence"] == "WA"]

    # city / zip null → "Unknown"
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Unknown")
    if "zip" in df.columns:
        df["zip"] = df["zip"].fillna("Unknown")

    # Standardize model column
    if "model" in df.columns:
        df["model"] = df["model"].map(lambda x: x.title() if isinstance(x, str) else x)

        # consolidate plug-in/electric suffixes
        repl = {
            "Niro Electric": "Niro",
            "Niro Plug-In Hybrid": "Niro",
            "Prius Plug-In": "Prius",
            "Prius Plug-In Hybrid": "Prius",
            "Kona Electric": "Kona",
            "Optima Plug-In Hybrid": "Optima",
            "Sonata Plug-In Hybrid": "Sonata",
            "Xc60 Awd Phev": "Xc60 Awd",
            "Xc90 Awd Phev": "Xc90 Awd"
        }
        df["model"] = df["model"].replace(repl)

    return df


@st.cache_data(show_spinner=True)
def get_ev_dataset():
    raw = load_ev_raw()
    cleaned = clean_ev_data(raw)

    # build EVs-on-the-road cumulative sum per county (Original Title only)
    county_dict = {}
    for county in cleaned["county"].unique():
        s = (cleaned[
            (cleaned["county"] == county) &
            (cleaned["transaction_type"] == "Original Title")
        ]
             .resample("M")
             .size()
             .cumsum())
        county_dict[county] = s

    df_cumsum = pd.DataFrame(county_dict).fillna(0.0)
    df_cumsum["State Total"] = df_cumsum.sum(axis=1)
    return cleaned, df_cumsum


@st.cache_data(show_spinner=True)
def load_charger_data():
    """
    Load pre-aggregated charger counts per county from
    df_charger_counts-07-13-2021.csv

    Expected columns (case-insensitive):
      - 'County'
      - 'Charger Count' (or similar like 'charger_count')
    """
    df = pd.read_csv(CHARGER_FILE)

    # Normalize column names a bit (strip, title-case for user display)
    original_cols = df.columns.tolist()

    # Try to detect the county and count columns in a robust way
    county_col = None
    count_col = None
    for c in original_cols:
        name = c.strip().lower().replace(" ", "_")
        if "county" == name or name.endswith("_county"):
            county_col = c
        if ("charger" in name or "station" in name) and ("count" in name or "total" in name):
            count_col = c

    # Fallbacks if above logic didn't find them
    if county_col is None:
        for c in original_cols:
            if "county" in c.lower():
                county_col = c
                break
    if count_col is None:
        for c in original_cols:
            if "count" in c.lower():
                count_col = c
                break

    if county_col is None or count_col is None:
        st.error(
            "Charger CSV should have County and Count columns. "
            f"Found columns: {original_cols}"
        )
        return pd.DataFrame(columns=["County", "Charger Count"])

    df = df.rename(columns={county_col: "County", count_col: "Charger Count"})
    df["County"] = df["County"].astype(str).str.strip()
    df["Charger Count"] = df["Charger Count"].astype(int)

    return df[["County", "Charger Count"]]


# -------------------------------------------------------------------
# MODELING HELPERS
# -------------------------------------------------------------------

def separate_data_by_county(county, df_cumsum):
    df_county = pd.DataFrame(df_cumsum[county])
    df_county.columns = ["EV's on the Road"]
    return df_county


def train_test_split_ts(df_county, train_size):
    n = len(df_county)
    split_idx = int(round(n * train_size, 0))
    train = df_county.iloc[:split_idx, 0]
    test = df_county.iloc[split_idx:, 0]
    return train, test


def plot_train_test(train, test, county):
    fig, ax = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    ax.set_title(f"Train/Test Split – {county} County")
    ax.set_xlabel("Date")
    ax.set_ylabel("EVs on the Road")
    ax.legend()
    st.pyplot(fig)


def get_forecast(model, train, test):
    fc = model.get_forecast(steps=len(test))
    conf_int = fc.conf_int()
    df = pd.DataFrame({
        "Lower CI": conf_int.iloc[:, 0],
        "Upper CI": conf_int.iloc[:, 1],
        "Forecasts": fc.predicted_mean
    })
    df.index = test.index
    return df


def plot_forecast(train, test, forecast_df, county):
    fig, ax = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax, color="black", label="Train")
    test.plot(ax=ax, color="purple", label="Actual (Test)")
    forecast_df["Forecasts"].plot(ax=ax, color="blue", linestyle="--", label="Forecast")
    ax.fill_between(forecast_df.index,
                    forecast_df["Lower CI"],
                    forecast_df["Upper CI"],
                    alpha=0.3)
    ax.set_title(f"Forecast vs Actual – {county} County")
    ax.set_xlabel("Date")
    ax.set_ylabel("EVs on the Road")
    ax.legend()
    st.pyplot(fig)


def get_prediction(model, full_df, steps):
    fc = model.get_forecast(steps=steps)
    conf_int = fc.conf_int()
    df = pd.DataFrame({
        "Lower CI": conf_int.iloc[:, 0],
        "Upper CI": conf_int.iloc[:, 1],
        "Predictions": fc.predicted_mean
    })
    return df


def plot_predictions(full_df, prediction_df, county):
    fig, ax = plt.subplots(figsize=(10, 4))
    full_df["EV's on the Road"].plot(ax=ax, label="Observed")
    prediction_df["Predictions"].plot(ax=ax, linestyle="--", label="Predicted")
    ax.fill_between(prediction_df.index,
                    prediction_df["Lower CI"],
                    prediction_df["Upper CI"],
                    alpha=0.3)
    ax.set_title(f"Predicted EVs on the Road – {county} County")
    ax.set_xlabel("Date")
    ax.set_ylabel("EVs on the Road")
    ax.legend()
    st.pyplot(fig)


# -------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Introduction",
        "State & County Trends",
        "Top Models by County",
        "Time Series Modeling",
        "Charger Infrastructure",
        "County Comparison & Recommendations",
        "Appendix – Predictions Table"
    ]
)

# Load core datasets once
with st.spinner("Loading and preparing datasets..."):
    ev_df_raw, df_cumsum = get_ev_dataset()
    charger_df = load_charger_data()

# Filter top-ten-only version of df_cumsum
df_cumsum_top10 = df_cumsum[TOP_TEN_COUNTIES + ["State Total"]]

# -------------------------------------------------------------------
# SECTION: INTRODUCTION
# -------------------------------------------------------------------
if section == "Introduction":
    st.header("🌍 Market at a Glance")

    st.markdown(
        """
        Transportation is the **largest source of U.S. greenhouse gas emissions**,  
        and EV adoption is rapidly accelerating, especially in states like **Washington**.

        In this app we:

        - Use **EV title & registration data** from Washington state  
        - Build **time-series models per county** for EV counts  
        - Combine that with **current charger counts**  
        - Identify **top counties to invest** in new charging stations  
        """
    )

    st.subheader("Business Problem & Goal")
    st.markdown(
        """
        > **Goal:** Identify Washington counties with **high projected EV demand**  
        > and **relatively low charger density**, to recommend **top investment targets**  
        > for EV charging companies.
        """
    )

    st.info(
        "Use the sidebar to move through the analysis – from raw data to final recommendations."
    )

# -------------------------------------------------------------------
# SECTION: STATE & COUNTY TRENDS
# -------------------------------------------------------------------
elif section == "State & County Trends":
    st.header("📈 Electric Vehicles on the Road – State & Counties")

    st.subheader("Statewide EV Count Over Time")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    df_cumsum_top10["State Total"].plot(ax=ax1)
    ax1.set_title("Electric Vehicles on the Road in Washington State")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("EV Count (Cumulative)")
    st.pyplot(fig1)

    st.markdown(
        "The statewide EV count has grown **rapidly and non-linearly** over the past decade."
    )

    st.subheader("Top 10 Counties – EV Count Over Time")
    st.caption("All top-ten counties (King dominates visually).")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for c in TOP_TEN_COUNTIES:
        df_cumsum_top10[c].plot(ax=ax2, label=c)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("EV Count")
    ax2.set_title("Total EV Count by County – Top 10")
    ax2.legend(loc="upper left", ncol=2, fontsize=8)
    st.pyplot(fig2)

    st.subheader("Top 9 Counties (Excluding King)")

    no_king = [c for c in TOP_TEN_COUNTIES if c != "King"]
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for c in no_king:
        df_cumsum_top10[c].plot(ax=ax3, label=c)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("EV Count")
    ax3.set_title("Total EV Count by County – Excluding King")
    ax3.legend(loc="upper left", ncol=2, fontsize=8)
    st.pyplot(fig3)

    st.markdown(
        """
        - **King County** (Seattle area) leads by a large margin.  
        - Among the rest, **Snohomish**, **Pierce**, and **Clark** show strong growth.
        """
    )

# -------------------------------------------------------------------
# SECTION: TOP MODELS BY COUNTY
# -------------------------------------------------------------------
elif section == "Top Models by County":
    st.header("🚗 Most Purchased EV Models by County")

    st.markdown(
        "We restrict to **Original Title** transactions (new EV purchases)."
    )

    df_original = ev_df_raw[ev_df_raw["transaction_type"] == "Original Title"]

    county_choice = st.selectbox("Select a county", TOP_TEN_COUNTIES)

    top_n = st.slider("Number of top models to display", 5, 20, 10)

    df_county = df_original[df_original["county"] == county_choice]
    model_counts = df_county["model"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=model_counts.values, y=model_counts.index, ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("Model")
    ax.set_title(f"Top {top_n} Models in {county_choice} County")
    st.pyplot(fig)

    st.markdown(
        """
        Across counties, you typically see **Nissan Leaf** and **Tesla Model 3**  
        at the top – reflecting both early mass-market EVs and popular newer models.
        """
    )

# -------------------------------------------------------------------
# SECTION: TIME SERIES MODELING
# -------------------------------------------------------------------
elif section == "Time Series Modeling":
    st.header("📊 Time Series Modeling – SARIMAX by County")

    st.markdown(
        """
        For each of the top 10 counties, we:

        1. Build a monthly time series of **EVs on the road**  
        2. Split into **train/test**  
        3. Fit a **SARIMAX model** with parameters tuned offline  
        4. Validate on the test set  
        5. Fit on full data and **predict into the future**
        """
    )

    county = st.selectbox("Select a county for modeling", TOP_TEN_COUNTIES)

    df_county = separate_data_by_county(county, df_cumsum_top10)

    st.subheader(f"Decomposition – Trend & Seasonality ({county})")

    decomp = tsa.seasonal_decompose(df_county, model="additive", period=12)
    fig_dec = decomp.plot()
    fig_dec.set_size_inches(10, 6)
    st.pyplot(fig_dec)

    # Train/Test
    st.subheader("Train/Test Split")
    train_size = TRAIN_SPLIT[county]
    train, test = train_test_split_ts(df_county, train_size)
    plot_train_test(train, test, county)

    # Fit SARIMAX on train
    st.subheader("Model Fit & Forecast vs Test")

    cfg = MODEL_CONFIG[county]
    order = cfg["order"]
    seasonal_order = cfg.get("seasonal_order", (0, 0, 0, 0))

    with st.spinner("Fitting SARIMAX model on training data..."):
        model_train = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_invertibility=False,
            enforce_stationarity=False
        ).fit(disp=False)

    forecast_df = get_forecast(model_train, train, test)
    plot_forecast(train, test, forecast_df, county)

    st.markdown(
        "The **test data generally stays within the forecast confidence interval**, "
        "indicating that the model captures the trend and seasonality reasonably well."
    )

    # Fit on full data and predict into future
    st.subheader("Predictions into the Future")

    steps = len(test)

    with st.spinner("Refitting model on full data and predicting..."):
        model_full = SARIMAX(
            df_county["EV's on the Road"],
            order=order,
            seasonal_order=seasonal_order,
            enforce_invertibility=False,
            enforce_stationarity=False
        ).fit(disp=False)

        pred_df = get_prediction(model_full, df_county, steps=steps)

    # Align prediction index to monthly periods just after the last observed date
    last_date = df_county.index[-1]
    pred_index = pd.date_range(start=last_date + pd.offsets.MonthEnd(1),
                               periods=steps,
                               freq="M")
    pred_df.index = pred_index

    plot_predictions(df_county, pred_df, county)

    st.markdown(
        f"""
        For **{county} County**, the model predicts that EV counts will **keep growing over the next {steps} months**  
        (often with an exponential-like pattern in the upper confidence bound).
        """
    )

# -------------------------------------------------------------------
# SECTION: CHARGER INFRASTRUCTURE
# -------------------------------------------------------------------
elif section == "Charger Infrastructure":
    st.header("🔌 Current Charger Infrastructure by County")

    st.markdown(
        """
        We use **pre-aggregated charger counts**  
        from `df_charger_counts-07-13-2021.csv`.
        """
    )

    if charger_df.empty:
        st.error("Charger data could not be loaded correctly.")
    else:
        df_charger_counts = charger_df.copy()

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.barplot(
            data=df_charger_counts.sort_values("Charger Count", ascending=False),
            x="Charger Count",
            y="County",
            ax=ax
        )
        ax.set_title("Charging Station Counts by County")
        ax.set_xlabel("# of Charging Stations")
        ax.set_ylabel("County")
        st.pyplot(fig)

        st.subheader("Raw Charger Count Table")
        st.dataframe(df_charger_counts)

# -------------------------------------------------------------------
# SECTION: COUNTY COMPARISON & RECOMMENDATIONS
# -------------------------------------------------------------------
elif section == "County Comparison & Recommendations":
    st.header("📊 County Comparison: EV Demand vs Charger Supply")

    if charger_df.empty:
        st.error("Charger data could not be loaded correctly.")
    else:
        df_charger_counts = charger_df.copy()

        # Build dictionary of county-specific EV series + predictions
        county_info = {}

        for c in TOP_TEN_COUNTIES:
            df_county = separate_data_by_county(c, df_cumsum_top10)
            train_size = TRAIN_SPLIT[c]
            train, test = train_test_split_ts(df_county, train_size)

            cfg = MODEL_CONFIG[c]
            order = cfg["order"]
            seasonal_order = cfg.get("seasonal_order", (0, 0, 0, 0))

            model_full = SARIMAX(
                df_county["EV's on the Road"],
                order=order,
                seasonal_order=seasonal_order,
                enforce_invertibility=False,
                enforce_stationarity=False
            ).fit(disp=False)

            steps = len(test)
            pred_df = get_prediction(model_full, df_county, steps)
            last_date = df_county.index[-1]
            pred_idx = pd.date_range(start=last_date + pd.offsets.MonthEnd(1),
                                     periods=steps,
                                     freq="M")
            pred_df.index = pred_idx

            county_info[c] = {
                "df": df_county,
                "pred": pred_df
            }

        # Comparison: last observed vs last predicted, and charger counts
        comparison_df = pd.DataFrame()

        for c in TOP_TEN_COUNTIES:
            row = {}
            row["County"] = c

            df_c = county_info[c]["df"]
            last_obs_val = df_c["EV's on the Road"].iloc[-1]
            row["EV Count Today (last observed)"] = int(round(last_obs_val, 0))

            pred = county_info[c]["pred"]
            future_ev_val = int(round(pred["Predictions"].iloc[-1], 0))
            row["EV Prediction (Future Horizon)"] = future_ev_val

            chargers = df_charger_counts.loc[
                df_charger_counts["County"] == c, "Charger Count"
            ]
            if len(chargers) == 0:
                charger_count = 0
            else:
                charger_count = int(chargers.iloc[0])
            row["Existing Charger Count"] = charger_count

            row["EVs Added (Obs → Future)"] = (
                row["EV Prediction (Future Horizon)"] - row["EV Count Today (last observed)"]
            )

            if charger_count > 0:
                row["EVs per Charger"] = round(
                    row["EV Prediction (Future Horizon)"] / charger_count, 1
                )
                row["Chargers per EV"] = round(
                    charger_count / row["EV Prediction (Future Horizon)"], 3
                )
            else:
                row["EVs per Charger"] = np.nan
                row["Chargers per EV"] = np.nan

            comparison_df = pd.concat([comparison_df, pd.DataFrame([row])],
                                      ignore_index=True)

        comparison_df = comparison_df.set_index("County")
        comparison_df = comparison_df.sort_values("EVs per Charger", ascending=False)

        st.subheader("County Comparison Table")

        st.dataframe(comparison_df.style.background_gradient(
            axis=0,
            subset=["EVs per Charger"],
            cmap="RdYlGn"
        ))

        st.subheader("EVs per Charger (Higher = More Demand per Station)")
        fig, ax = plt.subplots(figsize=(8, 4))
        comparison_df["EVs per Charger"].plot(kind="bar", ax=ax)
        ax.set_ylabel("EVs per Charger (Future)")
        ax.set_title("EV Demand Relative to Charger Supply")
        st.pyplot(fig)

        st.subheader("Recommendations")

        st.markdown(
            """
            Based on **projected EV counts** and **existing charger infrastructure**:

            **Top recommended counties to invest in new chargers:**

            1. **Clark County**  
            2. **Snohomish County**  
            3. **Whatcom County**

            These balance:

            - Strong **future EV growth**  
            - **Higher EVs-per-charger ratios** (more demand per station)  
            - Strategic location near **metros, airports, and highways**
            """
        )

# -------------------------------------------------------------------
# SECTION: APPENDIX – PREDICTIONS TABLE
# -------------------------------------------------------------------
elif section == "Appendix – Predictions Table":
    st.header("📑 Appendix: Monthly EV Count Predictions by County")

    county = st.selectbox("Select a county", TOP_TEN_COUNTIES)

    df_county = separate_data_by_county(county, df_cumsum_top10)
    train_size = TRAIN_SPLIT[county]
    train, test = train_test_split_ts(df_county, train_size)

    cfg = MODEL_CONFIG[county]
    order = cfg["order"]
    seasonal_order = cfg.get("seasonal_order", (0, 0, 0, 0))

    with st.spinner("Fitting model and generating predictions table..."):
        model_full = SARIMAX(
            df_county["EV's on the Road"],
            order=order,
            seasonal_order=seasonal_order,
            enforce_invertibility=False,
            enforce_stationarity=False
        ).fit(disp=False)

        steps = len(test)
        pred_df = get_prediction(model_full, df_county, steps)

    last_date = df_county.index[-1]
    pred_index = pd.date_range(start=last_date + pd.offsets.MonthEnd(1),
                               periods=steps,
                               freq="M")
    pred_df.index = pred_index

    st.markdown(f"**Predictions for {county} County (monthly horizon)**")
    st.dataframe(pred_df.round(0))

    csv_bytes = pred_df.round(0).to_csv().encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_bytes,
        file_name=f"{county}_ev_predictions.csv",
        mime="text/csv"
    )
