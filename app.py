import pathlib

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = pathlib.Path(__file__).parent / "data.csv"


@st.cache_data
def load_data(file_obj=None) -> pd.DataFrame:
    """
    Load and clean the commodity price data.

    - file_obj: can be a pathlib.Path, str, or an uploaded file-like object.
    """
    if file_obj is None:
        file_obj = DATA_PATH

    # Read CSV and parse dates (dayfirst because format is dd-mm-yyyy).
    df = pd.read_csv(
        file_obj,
        parse_dates=["Arrival_Date"],
        dayfirst=True,
    )

    # Basic cleaning
    # Drop rows with missing core fields
    core_cols = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade",
        "Arrival_Date",
        "Min_Price",
        "Max_Price",
        "Modal_Price",
    ]
    existing_core_cols = [c for c in core_cols if c in df.columns]
    df = df.dropna(subset=existing_core_cols)

    # Ensure numeric price columns
    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with non-positive or missing modal price (main metric)
    if "Modal_Price" in df.columns:
        df = df[df["Modal_Price"] > 0]

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

    return df_filtered, price_metric, analysis_type


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


def show_price_trends(df: pd.DataFrame, price_col: str) -> None:
    st.subheader(f"Price trends over time ({price_col})")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # Aggregate by date, commodity, and market to smooth large datasets if needed
    grouped = (
        df.groupby(["Arrival_Date", "Commodity", "Market"], as_index=False)[price_col]
        .mean()
    )

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
    data_source = st.sidebar.radio(
        "Choose data source",
        options=["Use bundled data.csv", "Upload CSV file"],
    )

    df = None
    if data_source == "Use bundled data.csv":
        if not DATA_PATH.exists():
            st.error(
                f"Expected data file not found at `{DATA_PATH}`. "
                "Please switch to 'Upload CSV file' and provide the dataset."
            )
            return
        df = load_data(DATA_PATH)
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.info("Please upload a CSV file to continue.")
            return
        df = load_data(uploaded)

    if df is None or df.empty:
        st.error("Loaded dataset is empty or invalid.")
        return

    with st.spinner("Applying filters..."):
        df_filtered, price_metric, analysis_type = apply_filters(df)

    if df_filtered.empty:
        st.warning("No data matches your filters. Try broadening your selection.")
        return

    show_overview(df_filtered)

    if analysis_type == "Price trends over time":
        show_price_trends(df_filtered, price_metric)
    elif analysis_type == "Market comparison":
        show_market_comparison(df_filtered, price_metric)
    elif analysis_type == "Summary statistics":
        show_summary_stats(df_filtered, price_metric)
    elif analysis_type == "Price distribution":
        show_distributions(df_filtered, price_metric)


if __name__ == "__main__":
    main()

