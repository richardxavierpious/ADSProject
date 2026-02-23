import pathlib
import json
import time

import gdown
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = pathlib.Path(__file__).parent / "data.csv"
# Public Google Drive file details for the full dataset
DRIVE_FILE_ID = "1GokpTUtpDDqC0v_Toj04ZKk0OH5k8dGn"
DATA_URL = (
    "https://drive.google.com/uc?export=download&"
    f"id={DRIVE_FILE_ID}"
)
LOG_PATH = pathlib.Path("debug-8f4dcb.log")


def _debug_log(hypothesis_id: str, message: str, data: dict | None = None, run_id: str = "pre-fix") -> None:
    """Append a single NDJSON log line for debugging hypotheses."""
    ts = int(time.time() * 1000)
    entry = {
        "sessionId": "8f4dcb",
        "id": f"log_{ts}",
        "timestamp": ts,
        "location": "app.py",
        "message": message,
        "data": data or {},
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        # Logging must never break the app
        pass


@st.cache_data
def load_data(file_obj=None) -> pd.DataFrame:
    """
    Load and clean the commodity price data.

    Cleaning steps:
    - Remove exact duplicate rows
    - Handle missing values
    - Remove extreme outliers in price columns

    - file_obj: can be a pathlib.Path, str, or an uploaded file-like object.
    """
    if file_obj is None:
        file_obj = DATA_PATH

    # region agent log
    _debug_log(
        hypothesis_id="H1",
        message="load_data called",
        data={"file_obj_type": str(type(file_obj)), "file_obj_str": str(file_obj)[:200]},
    )
    # endregion

    # If we're loading from Google Drive, use gdown to properly handle
    # large-file confirmation pages and download the actual CSV content.
    if isinstance(file_obj, str) and file_obj == DATA_URL:
        drive_target = pathlib.Path("drive_data.csv")
        try:
            gdown.download(id=DRIVE_FILE_ID, output=str(drive_target), quiet=True)
            file_obj = drive_target
            # region agent log
            _debug_log(
                hypothesis_id="H4",
                message="Downloaded data from Google Drive using gdown",
                data={"output_path": str(drive_target)},
            )
            # endregion
        except Exception as exc:
            # region agent log
            _debug_log(
                hypothesis_id="H4",
                message="gdown download failed",
                data={"error": str(exc)},
            )
            # endregion
            raise

    # Read CSV first without date parsing so we can handle
    # missing/renamed columns more gracefully (especially for remote URLs).
    df = pd.read_csv(file_obj)

    # region agent log
    _debug_log(
        hypothesis_id="H2",
        message="DataFrame loaded",
        data={
            "shape": list(df.shape),
            "columns": list(map(str, df.columns)),
        },
    )
    # endregion

    # Normalise column names a bit
    df.columns = df.columns.str.strip()

    if "Arrival_Date" in df.columns:
        df["Arrival_Date"] = pd.to_datetime(
            df["Arrival_Date"],
            dayfirst=True,
            errors="coerce",
        )
    else:
        raise ValueError("Expected column 'Arrival_Date' not found in dataset.")

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Ensure numeric price columns
    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values:
    # - Require key identifiers and modal price
    required_cols = [c for c in [
        "State",
        "District",
        "Market",
        "Commodity",
        "Arrival_Date",
        "Modal_Price",
    ] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    # - Fill other categorical columns with a placeholder
    for col in ["Variety", "Grade"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Remove rows with non-positive modal price (main metric)
    if "Modal_Price" in df.columns:
        df = df[df["Modal_Price"] > 0]

    # Domain-based rule: prices are rupees per 100kg.
    # Assume realistic minimum is 1 rupee per kg -> 100 per 100kg.
    # Treat values below 100 as outliers and drop those rows.
    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        if col in df.columns:
            df = df[df[col].isna() | (df[col] >= 100)]

    # Remove upper outlier rows in price columns using IQR rule
    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        if col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            upper = q3 + 1.5 * iqr
            df = df[
                df[col].isna()
                | (df[col] <= upper)
            ]

    # Add helper time columns
    if "Arrival_Date" in df.columns:
        df["Year"] = df["Arrival_Date"].dt.year
        df["Month"] = df["Arrival_Date"].dt.month

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters and return the filtered dataframe."""
    st.sidebar.header("Filters")

    # Commodity filter
    commodities = sorted(df["Commodity"].dropna().unique())
    selected_commodities = st.sidebar.multiselect(
        "Commodity",
        options=commodities,
        default=commodities[:5] if len(commodities) > 5 else commodities,
    )

    # Market filter
    markets = sorted(df["Market"].dropna().unique())
    selected_markets = st.sidebar.multiselect(
        "Market",
        options=markets,
        default=markets,
    )

    # District filter
    districts = sorted(df["District"].dropna().unique())
    selected_districts = st.sidebar.multiselect(
        "District",
        options=districts,
        default=districts,
    )

    # Date range filter
    min_date = df["Arrival_Date"].min().date()
    max_date = df["Arrival_Date"].max().date()
    date_range = st.sidebar.date_input(
        "Arrival date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    # Price metric
    price_metric = st.sidebar.selectbox(
        "Price metric",
        options=["Modal_Price", "Min_Price", "Max_Price"],
        index=0,
    )

    # Trend aggregation level (for time series)
    trend_freq = st.sidebar.selectbox(
        "Trend aggregation",
        options=["Daily", "Weekly", "Monthly"],
        index=2,
    )

    # Analysis type
    analysis_type = st.sidebar.radio(
        "Analysis to perform",
        options=[
            "Price trends over time",
            "Market comparison",
            "Summary statistics",
            "Price distribution",
        ],
    )

    # Apply filters to dataframe
    df_filtered = df.copy()

    if selected_commodities:
        df_filtered = df_filtered[df_filtered["Commodity"].isin(selected_commodities)]

    if selected_markets:
        df_filtered = df_filtered[df_filtered["Market"].isin(selected_markets)]

    if selected_districts:
        df_filtered = df_filtered[df_filtered["District"].isin(selected_districts)]

    df_filtered = df_filtered[
        (df_filtered["Arrival_Date"].dt.date >= start_date)
        & (df_filtered["Arrival_Date"].dt.date <= end_date)
    ]

    return df_filtered, price_metric, analysis_type, trend_freq


def show_overview(df: pd.DataFrame) -> None:
    """Display high-level overview stats for the filtered data."""
    st.subheader("Overview of filtered data")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Commodities", df["Commodity"].nunique())
    col3.metric("Markets", df["Market"].nunique())
    col4.metric("Districts", df["District"].nunique())

    with st.expander("Preview filtered data"):
        st.dataframe(df.head(500))


def show_price_trends(df: pd.DataFrame, price_col: str, freq: str) -> None:
    st.subheader(f"Price trends over time ({price_col})")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # Aggregate over time to reduce the number of plot points
    if freq == "Daily":
        grouped = (
            df.groupby(["Arrival_Date", "Commodity", "Market"], as_index=False)[price_col]
            .mean()
        )
    else:
        freq_map = {"Weekly": "W", "Monthly": "M"}
        resampled = (
            df.set_index("Arrival_Date")
            .groupby(["Commodity", "Market"])[price_col]
            .resample(freq_map.get(freq, "M"))
            .mean()
            .reset_index()
        )
        # Use a common x-axis column name
        resampled = resampled.rename(columns={"Arrival_Date": "Period"})
        grouped = resampled.rename(columns={"Period": "Arrival_Date"})

    fig = px.line(
        grouped,
        x="Arrival_Date",
        y=price_col,
        color="Commodity",
        line_group="Market",
        hover_data=["Market"],
        markers=True,
        labels={
            "Arrival_Date": "Date",
            price_col: "Price",
        },
    )
    fig.update_layout(legend_title_text="Commodity")
    st.plotly_chart(fig, use_container_width=True)


def show_market_comparison(df: pd.DataFrame, price_col: str) -> None:
    st.subheader(f"Market comparison ({price_col})")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    agg = (
        df.groupby(["Market", "Commodity"], as_index=False)[price_col]
        .mean()
        .sort_values(price_col, ascending=False)
    )

    fig = px.bar(
        agg,
        x="Market",
        y=price_col,
        color="Commodity",
        labels={"Market": "Market", price_col: "Average price"},
        barmode="group",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def show_summary_stats(df: pd.DataFrame, price_col: str) -> None:
    st.subheader(f"Summary statistics by commodity and market ({price_col})")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    stats = (
        df.groupby(["Commodity", "Market"])[price_col]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .reset_index()
        .rename(columns={"count": "observations"})
    )

    st.dataframe(stats)

    csv = stats.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download summary as CSV",
        data=csv,
        file_name="summary_stats.csv",
        mime="text/csv",
    )


def show_distributions(df: pd.DataFrame, price_col: str) -> None:
    st.subheader(f"Price distribution ({price_col})")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    tab1, tab2 = st.tabs(["By commodity", "Histogram"])

    with tab1:
        fig_box = px.box(
            df,
            x="Commodity",
            y=price_col,
            color="Commodity",
            labels={"Commodity": "Commodity", price_col: "Price"},
        )
        fig_box.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        fig_hist = px.histogram(
            df,
            x=price_col,
            nbins=40,
            labels={price_col: "Price"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Kerala Commodity Prices Dashboard",
        layout="wide",
        page_icon="📊",
    )

    st.title("Kerala Commodity Prices Dashboard")
    st.markdown(
        "Explore prices of various commodities across markets in Kerala. "
        "Use the sidebar to choose filters and analyses."
    )

    # Data source selection
    st.sidebar.header("Data source")

    # Build options dynamically so local development can still use a local file,
    # while deployed apps can simply use the Google Drive-hosted CSV.
    data_source_options = []
    if DATA_PATH.exists():
        data_source_options.append("Use local data.csv")
    data_source_options.append("Use Google Drive data.csv")
    data_source_options.append("Upload CSV file")

    data_source = st.sidebar.radio(
        "Choose data source",
        options=data_source_options,
    )

    df = None
    try:
        # region agent log
        _debug_log(
            hypothesis_id="H3",
            message="Selected data source",
            data={"data_source": data_source},
        )
        # endregion
        if data_source == "Use local data.csv":
            df = load_data(DATA_PATH)
        elif data_source == "Use Google Drive data.csv":
            df = load_data(DATA_URL)
        else:
            uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
            if uploaded is None:
                st.info("Please upload a CSV file to continue.")
                return
            df = load_data(uploaded)
    except ValueError as e:
        st.error(f"Error while loading data: {e}")
        return

    if df is None or df.empty:
        st.error("Loaded dataset is empty or invalid.")
        return

    with st.spinner("Applying filters..."):
        df_filtered, price_metric, analysis_type, trend_freq = apply_filters(df)

    if df_filtered.empty:
        st.warning("No data matches your filters. Try broadening your selection.")
        return

    show_overview(df_filtered)

    if analysis_type == "Price trends over time":
        show_price_trends(df_filtered, price_metric, trend_freq)
    elif analysis_type == "Market comparison":
        show_market_comparison(df_filtered, price_metric)
    elif analysis_type == "Summary statistics":
        show_summary_stats(df_filtered, price_metric)
    elif analysis_type == "Price distribution":
        show_distributions(df_filtered, price_metric)


if __name__ == "__main__":
    main()

