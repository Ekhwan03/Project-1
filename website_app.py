from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from app import (
    classify_aqi,
    forecast_future_aqi,
    get_aqi_summary,
    get_forecast_summary,
    get_recommendation,
    initialise_database,
    load_dataset,
    train_prediction_model,
)


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="AeroSense | Air Quality System",
    page_icon="🌿",
    layout="wide",
)

components.html(
    """
    <script>
        function setupHeaderScrollBehavior() {
            const header = document.querySelector('header[data-testid="stHeader"]');
            if (!header) return;

            let lastScrollY = window.scrollY || window.pageYOffset || 0;

            function updateHeader() {
                const currentY = window.scrollY || window.pageYOffset || 0;
                const shouldHide = currentY > lastScrollY && currentY > 24;

                header.style.transform = shouldHide ? 'translateY(-120%)' : 'translateY(0)';
                header.style.opacity = shouldHide ? '0' : '1';
                header.style.pointerEvents = shouldHide ? 'none' : 'auto';
                lastScrollY = currentY;
            }

            header.style.transition = 'transform 0.25s ease, opacity 0.25s ease';
            header.style.willChange = 'transform, opacity';
            header.style.position = 'sticky';
            header.style.top = '0';
            header.style.zIndex = '1000';

            window.addEventListener('scroll', updateHeader, { passive: true });
            updateHeader();
        }

        window.addEventListener('load', setupHeaderScrollBehavior);
        setTimeout(setupHeaderScrollBehavior, 300);
        setTimeout(setupHeaderScrollBehavior, 800);
    </script>
    """,
    height=0,
    scrolling=False,
)

st.markdown(
    """
    <style>
        :root {
            --app-bg: #f5faf8;
            --panel-bg: #ffffff;
            --sidebar-bg: #eaf5f1;
            --text-color: #123b37;
            --muted-color: #3a5f5b;
            --border-color: #dfeae6;
            --success-bg: #edf9f5;
            --success-text: #123b37;
            --banner-bg: linear-gradient(135deg, #164E46 0%, #1c6b62 100%);
            --banner-text: #ffffff;
        }

        html[data-theme="light"], html[data-theme="light"] body,
        html[data-theme="dark"], html[data-theme="dark"] body {
            background: transparent;
        }

        html[data-theme="light"] {
            --app-bg: #f5faf8;
            --panel-bg: #ffffff;
            --sidebar-bg: #eaf5f1;
            --text-color: #123b37;
            --muted-color: #3a5f5b;
            --border-color: #dfeae6;
            --success-bg: #edf9f5;
            --success-text: #123b37;
            --banner-bg: linear-gradient(135deg, #164E46 0%, #1c6b62 100%);
            --banner-text: #ffffff;
        }

        html[data-theme="dark"] {
            --app-bg: #0b1220;
            --panel-bg: #15232d;
            --sidebar-bg: #111c23;
            --text-color: #edf7f5;
            --muted-color: #d0e7e1;
            --border-color: #2b3d45;
            --success-bg: rgba(29, 86, 82, 0.9);
            --success-text: #edf7f5;
            --banner-bg: linear-gradient(135deg, #12352f 0%, #1c6b62 100%);
            --banner-text: #ffffff;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: none !important;
            position: sticky !important;
            top: 0 !important;
            z-index: 1000 !important;
            transition: transform 0.25s ease, opacity 0.25s ease !important;
        }

        header[data-testid="stHeader"] * {
            color: var(--text-color) !important;
        }

        .stAppToolbar {
            background: transparent !important;
        }

        .stAppToolbar button,
        .stAppToolbar [data-testid="stBaseButton-header"] {
            width: 2.4rem !important;
            height: 2.4rem !important;
            min-width: 2.4rem !important;
            border-radius: 0.75rem !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            background: linear-gradient(135deg, rgba(14, 69, 62, 0.98) 0%, rgba(22, 132, 119, 0.96) 100%) !important;
            box-shadow: 0 8px 20px rgba(10, 52, 46, 0.18) !important;
            padding: 0 !important;
            color: #ffffff !important;
        }

        .stAppToolbar button:hover,
        .stAppToolbar button:focus-visible,
        .stAppToolbar [data-testid="stBaseButton-header"]:hover,
        .stAppToolbar [data-testid="stBaseButton-header"]:focus-visible {
            background: linear-gradient(135deg, rgba(20, 82, 74, 0.98) 0%, rgba(29, 141, 127, 0.96) 100%) !important;
            border-color: rgba(255, 255, 255, 0.24) !important;
            color: #ffffff !important;
            box-shadow: 0 10px 22px rgba(10, 52, 46, 0.24) !important;
        }

        .stAppToolbar button span[data-testid="stIconMaterial"] {
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 1.5rem !important;
            height: 1.5rem !important;
            color: #ffffff !important;
            font-size: 1.35rem !important;
            line-height: 1 !important;
            visibility: hidden !important;
        }

        .stAppToolbar button span[data-testid="stIconMaterial"]::before {
            content: "menu";
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: "Material Symbols Rounded";
            font-size: 1.35rem;
            font-weight: 400;
            line-height: 1;
            visibility: visible;
            color: #ffffff;
        }

        .stApp {
            background: var(--app-bg) !important;
            color: var(--text-color) !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            color: var(--text-color) !important;
        }

        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        p, li, span, div,
        .stMarkdown, .stTextInput, .stSelectbox,
        .stNumberInput, .stDateInput, .stRadio,
        .stCheckbox, .stButton, .stDataFrame,
        .stTabs, .stTab, .stExpander {
            color: var(--text-color) !important;
        }

        [data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(22, 132, 119, 0.06);
            color: var(--text-color) !important;
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar-bg) !important;
            border-right: 1px solid #cfe2db;
            color: var(--text-color) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-color) !important;
        }

        [data-testid="stSidebar"] .stRadio {
            color: var(--text-color) !important;
            font-weight: 600;
        }

        [data-testid="stBaseSelectbox"],
        [data-testid="stSelectbox"],
        [data-testid="stBaseSelectbox"] * {
            color: #ffffff !important;
        }

        div[role="listbox"],
        div[role="listbox"] * {
            color: #ffffff !important;
            background: #123b37 !important;
        }

        button[kind="elementToolbar"],
        button[data-testid="stBaseButton-elementToolbar"],
        [data-testid="stElementToolbarButtonContainer"] {
            background: linear-gradient(135deg, rgba(14, 69, 62, 0.98) 0%, rgba(22, 132, 119, 0.96) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            box-shadow: 0 8px 20px rgba(10, 52, 46, 0.18) !important;
            border-radius: 10px !important;
            color: #ffffff !important;
        }

        [data-testid="stElementToolbarButtonContainer"] button,
        [data-testid="stElementToolbarButtonContainer"] svg,
        button[kind="elementToolbar"] svg,
        button[data-testid="stBaseButton-elementToolbar"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
            background: transparent !important;
            border: none !important;
        }

        [data-testid="stElementToolbarButtonContainer"] button:hover,
        button[kind="elementToolbar"]:hover,
        button[data-testid="stBaseButton-elementToolbar"]:hover {
            background: rgba(255, 255, 255, 0.12) !important;
            border-radius: 8px !important;
        }

        div.stButton > button {
            color: #123b37 !important;
            background: #ffffff !important;
            border: 1px solid #dfeae6 !important;
        }

        div.stButton > button span,
        div.stButton > button div,
        div.stButton > button p,
        div.stButton > button * {
            background: transparent !important;
            color: inherit !important;
            border: none !important;
        }

        div.stButton > button:hover,
        button[kind="header"]:hover {
            background: #edf9f5 !important;
            border-color: #b7d8d2 !important;
            color: #123b37 !important;
        }

        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="baseButton-primary"],
        button[kind="header"][variant="primary"],
        .stButton > button:focus-visible {
            background: #168477 !important;
            border-color: #168477 !important;
            color: #ffffff !important;
        }

        div.stDownloadButton > button,
        div.stDownloadButton > button span,
        div.stDownloadButton > button p,
        div.stDownloadButton > button div,
        div.stDownloadButton > button * {
            color: #123b37 !important;
            background: #ffffff !important;
            border-color: rgba(22, 132, 119, 0.25) !important;
        }

        div.stDownloadButton > button:hover {
            background: #edf9f5 !important;
            border-color: rgba(22, 132, 119, 0.4) !important;
            color: #123b37 !important;
        }

        div[data-testid="stHorizontalBlock"]:first-of-type {
            background: linear-gradient(135deg, rgba(14, 69, 62, 0.98) 0%, rgba(22, 132, 119, 0.96) 100%);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 18px;
            padding: 0.5rem;
            margin: 0 0 1.2rem 0;
            box-shadow: 0 12px 26px rgba(10, 52, 46, 0.18);
            backdrop-filter: blur(8px);
        }

        div[data-testid="stHorizontalBlock"]:first-of-type > div {
            padding: 0 0.15rem;
        }

        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button {
            width: 100%;
            min-height: 2.8rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background: rgba(255, 255, 255, 0.06) !important;
            color: #ffffff !important;
            font-weight: 700;
            letter-spacing: 0.01em;
            transition: all 0.2s ease;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03);
            white-space: nowrap;
            overflow: visible;
            text-overflow: clip;
        }

        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button:hover {
            background: rgba(255, 255, 255, 0.12) !important;
            border-color: rgba(255, 255, 255, 0.18) !important;
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(7, 32, 28, 0.12);
        }

        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button > div,
        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button span,
        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button p,
        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button div p {
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
            display: inline-block !important;
            line-height: 1.4 !important;
            vertical-align: middle !important;
            color: #ffffff !important;
        }

        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #1db8a8 0%, #0f8c7e 100%) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
            box-shadow: 0 10px 22px rgba(16, 110, 101, 0.24);
        }

        div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, #1aa89a 0%, #0e7f72 100%) !important;
            color: #ffffff !important;
        }

        .status-box {
            background: linear-gradient(135deg, #edf9f5 0%, #e9f2ef 100%);
            border-left: 4px solid #168477;
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin: 0.7rem 0 1.2rem 0;
            color: var(--success-text) !important;
        }

        .section-card {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .welcome-banner {
            background: var(--banner-bg);
            color: var(--banner-text);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(22, 78, 70, 0.12);
        }

        .welcome-banner .title {
            color: var(--banner-text);
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }

        .welcome-banner .subtitle {
            color: rgba(255,255,255,0.85);
            font-size: 1rem;
            margin-top: 0.35rem;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--app-bg) !important;
            color: var(--text-color) !important;
        }

        [data-testid="stMain"] {
            background: var(--app-bg) !important;
            color: var(--text-color) !important;
        }

        .stDataFrame {
            background: var(--panel-bg) !important;
            color: var(--text-color) !important;
        }

        .stDataFrameContainer {
            background: var(--panel-bg) !important;
            color: var(--text-color) !important;
        }

        .stDataFrame > div {
            color: var(--text-color) !important;
        }

        section[role="dialog"],
        section[role="dialog"] *,
        div[role="dialog"],
        div[role="dialog"] *,
        .st-emotion-cache-cpuwpc,
        .st-emotion-cache-cpuwpc * {
            color: #123b37 !important;
            background: #ffffff !important;
        }

        div[role="menu"],
        div[data-testid="stMainMenuList"],
        div[role="menu"] *,
        div[data-testid="stMainMenuList"] *,
        .stMainMenuPopover,
        .stMainMenuPopover * {
            background: #ffffff !important;
            color: #111827 !important;
            border-color: #dfeae6 !important;
        }

        button[role="menuitem"],
        button[role="menuitemradio"],
        button[role="menuitemcheckbox"],
        [data-testid^="stMainMenuItem-"] {
            background: transparent !important;
            color: #111827 !important;
            border: none !important;
        }

        button[role="menuitem"]:hover,
        button[role="menuitemradio"]:hover,
        button[role="menuitemcheckbox"]:hover,
        [data-testid^="stMainMenuItem-"]:hover,
        button[role="menuitemradio"][aria-checked="true"],
        button[role="menuitemcheckbox"][aria-checked="true"] {
            background: #edf9f5 !important;
            color: #0f172a !important;
        }

        span[data-testid="stMainMenuItemLabel"],
        kbd,
        div[role="menu"] span,
        div[role="menu"] label,
        div[role="menu"] button {
            color: #111827 !important;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg: #0b1220;
                --panel-bg: #15232d;
                --sidebar-bg: #111c23;
                --text-color: #edf7f5;
                --muted-color: #d0e7e1;
                --border-color: #2b3d45;
                --success-bg: rgba(29, 86, 82, 0.9);
                --success-text: #edf7f5;
                --banner-bg: linear-gradient(135deg, #12352f 0%, #1c6b62 100%);
            }

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                background: var(--app-bg) !important;
                color: var(--text-color) !important;
            }

            [data-testid="stMetric"] {
                background: var(--panel-bg) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-color) !important;
            }

            [data-testid="stSidebar"] {
                background: var(--sidebar-bg) !important;
                border-right: 1px solid #24333a !important;
            }

            [data-testid="stSidebar"] * {
                color: var(--text-color) !important;
            }

            .status-box {
                background: linear-gradient(135deg, rgba(29, 86, 82, 0.9) 0%, rgba(19, 51, 55, 0.95) 100%) !important;
                border-left: 4px solid #46d4bd !important;
                color: var(--text-color) !important;
            }

            div[data-testid="stHorizontalBlock"]:first-of-type {
                background: linear-gradient(135deg, rgba(21, 35, 45, 0.94) 0%, rgba(17, 28, 35, 0.96) 100%) !important;
                border-color: rgba(70, 212, 189, 0.2) !important;
                box-shadow: 0 12px 28px rgba(3, 7, 18, 0.3) !important;
            }

            div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button {
                background: rgba(255, 255, 255, 0.03) !important;
                border-color: rgba(255, 255, 255, 0.07) !important;
                color: #eaf7f5 !important;
            }

            div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button:hover {
                background: rgba(70, 212, 189, 0.08) !important;
                border-color: rgba(70, 212, 189, 0.25) !important;
                color: #ffffff !important;
            }

            div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button[data-testid="stBaseButton-primary"] {
                background: linear-gradient(135deg, #20b5a5 0%, #1d8d7f 100%) !important;
                border-color: rgba(126, 240, 211, 0.8) !important;
                color: #ffffff !important;
                box-shadow: 0 10px 22px rgba(32, 181, 165, 0.28) !important;
            }

            [data-testid="stElementToolbarButtonContainer"] {
                background: linear-gradient(135deg, rgba(20, 34, 40, 0.98) 0%, rgba(28, 70, 66, 0.96) 100%) !important;
                border-color: rgba(126, 240, 211, 0.28) !important;
            }

            [data-testid="stElementToolbarButtonContainer"] button,
            [data-testid="stElementToolbarButtonContainer"] svg,
            button[kind="elementToolbar"] svg,
            button[data-testid="stBaseButton-elementToolbar"] svg {
                color: #ffffff !important;
                fill: #ffffff !important;
            }

            .stDataFrame,
            .stDataFrameContainer {
                background: var(--panel-bg) !important;
                color: var(--text-color) !important;
            }

            section[role="dialog"],
            section[role="dialog"] *,
            div[role="dialog"],
            div[role="dialog"] *,
            .st-emotion-cache-cpuwpc,
            .st-emotion-cache-cpuwpc * {
                color: #edf7f5 !important;
                background: #15232d !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dashboard_data():
    initialise_database()
    df = load_dataset()
    summary = get_aqi_summary()
    metrics = train_prediction_model()
    return df, summary, metrics


def render_dashboard(df: pd.DataFrame, summary: dict) -> None:
    st.subheader("Dashboard Overview")

    selected_year = st.selectbox("Select year", ["All", "2023", "2024", "2025"])
    display_df = df if selected_year == "All" else df[df["Date"].dt.year == int(selected_year)].reset_index(drop=True)

    latest = display_df.sort_values("Date").tail(1).iloc[0]
    latest_aqi = int(latest["AQI"])
    avg_aqi = float(display_df["AQI"].mean())
    peak_aqi = int(display_df["AQI"].max())
    peak_location = display_df.loc[display_df["AQI"].idxmax(), "Location"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest AQI", f"{latest_aqi}", help="Latest recorded AQI in the current selection.")
    col2.metric("Average AQI", f"{avg_aqi:.2f}")
    col3.metric("Peak AQI", f"{peak_aqi}")
    col4.metric("Peak Location", peak_location)

    st.caption(
        f"Location focus: {peak_location} | "
        f"Records: {len(display_df):,} | "
        f"Latest date: {pd.to_datetime(latest['Date']).date()}"
    )

    st.divider()
    st.subheader("Historical Air Quality Trend")

    selected_location = st.selectbox("Select location", ["All"] + sorted(display_df["Location"].unique().tolist()))
    selected_pollutant = st.selectbox("Select pollutant", ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"])

    filtered = display_df if selected_location == "All" else display_df[display_df["Location"] == selected_location]
    trend_df = filtered.groupby("Date")["AQI"].mean().reset_index()
    fig = px.line(
        trend_df,
        x="Date",
        y="AQI",
        markers=True,
        labels={"Date": "Date", "AQI": "AQI"},
        template="plotly_white",
    )
    fig.update_traces(line=dict(color="#168477", width=3), marker=dict(size=6))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    pollutant_df = filtered.groupby("Date")[selected_pollutant].mean().reset_index()
    pollutant_fig = px.line(
        pollutant_df,
        x="Date",
        y=selected_pollutant,
        markers=True,
        template="plotly_white",
        labels={"Date": "Date", selected_pollutant: selected_pollutant},
    )
    pollutant_fig.update_traces(line=dict(color="#1E917A", width=3), marker=dict(size=6))
    pollutant_fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    st.plotly_chart(pollutant_fig, use_container_width=True)

    st.subheader("Latest Records")
    st.dataframe(
        filtered.sort_values("Date", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )


def render_trend_analysis(df: pd.DataFrame) -> None:
    st.subheader("Trend Analysis")

    selected_year = st.selectbox("Select year", ["All", "2023", "2024", "2025"])
    filtered_df = df if selected_year == "All" else df[df["Date"].dt.year == int(selected_year)].reset_index(drop=True)

    selected_location = st.selectbox("Select location", ["All"] + sorted(filtered_df["Location"].unique().tolist()))
    selected_pollutant = st.selectbox("Select pollutant", ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"])
    period = st.selectbox("Group data by", ["Monthly", "Yearly"])

    filtered = filtered_df if selected_location == "All" else filtered_df[filtered_df["Location"] == selected_location]
    trend_data = filtered.set_index("Date")[selected_pollutant]
    if period == "Monthly":
        trend_data = trend_data.resample("MS").mean().dropna().reset_index()
    else:
        trend_data = trend_data.resample("YS").mean().dropna().reset_index()

    trend_data.columns = ["date", "value"]
    if trend_data.empty:
        st.info("Not enough data for this trend view.")
    else:
        fig = px.line(
            trend_data,
            x="date",
            y="value",
            markers=True,
            template="plotly_white",
            labels={"date": "Period", "value": f"Average {selected_pollutant}"},
        )
        fig.update_traces(line=dict(color="#168477", width=3))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(trend_data.rename(columns={"date": "Period", "value": f"Average {selected_pollutant}"}), use_container_width=True, hide_index=True)


def render_smart_alert(df: pd.DataFrame) -> None:
    st.subheader("Smart Air Quality Alert")
    st.info(
        "Set a project-specific alert threshold below. "
        "This is not automatically an official health standard."
    )

    threshold = st.number_input("Alert threshold (AQI)", min_value=0.0, value=100.0, step=1.0)
    latest = df.sort_values("Date").tail(1).iloc[0]
    current_aqi = int(latest["AQI"])

    st.metric("Latest observed AQI", f"{current_aqi}")

    if current_aqi >= threshold:
        st.error(
            "Alert triggered: the latest observed value meets or exceeds your selected threshold."
        )
    else:
        st.success("No alert triggered under the selected threshold.")

    st.caption(
        f"Observation date: {pd.to_datetime(latest['Date']).date()} | "
        f"Location: {latest['Location']}"
    )
    st.write(get_recommendation(current_aqi))


def render_data_explorer(df: pd.DataFrame) -> None:
    st.subheader("Explore Dataset")

    search = st.text_input("Search location")
    table_data = df.copy()

    if search:
        table_data = table_data[
            table_data["Location"].str.contains(search, case=False, na=False)
        ]

    st.write(f"Matching records: {len(table_data):,}")
    st.dataframe(
        table_data.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = table_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv_data,
        file_name="air_quality_data.csv",
        mime="text/csv",
    )


def render_prediction(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Machine Learning Prediction")

    st.write(
        "This page uses the trained AQI prediction model to estimate upcoming haze conditions in the selected location."
    )

    selected_year = st.selectbox("Select forecast year", ["2023", "2024", "2025"])
    location = st.selectbox("Target location", sorted(df[df["Date"].dt.year == int(selected_year)]["Location"].unique().tolist()))
    days = st.slider("Forecast horizon (days)", min_value=3, max_value=30, value=7)
    forecast = forecast_future_aqi(location, days=days, selected_year=selected_year)

    if forecast.empty:
        st.warning(f"No forecast is available for {location}.")
        return

    avg_forecast = float(forecast["Forecasted_AQI"].mean())
    peak_row = forecast.loc[forecast["Forecasted_AQI"].idxmax()]
    peak_status = classify_aqi(float(peak_row["Forecasted_AQI"]))

    col1, col2, col3 = st.columns(3)
    col1.metric("Average forecast AQI", f"{avg_forecast:.1f}")
    col2.metric("Peak haze day", f"{peak_row['Forecasted_AQI']:.1f}")
    col3.metric("Peak status", peak_status)

    st.info(get_forecast_summary(forecast, location))

    fig = px.line(
        forecast,
        x="Date",
        y="Forecasted_AQI",
        markers=True,
        template="plotly_white",
        labels={"Date": "Date", "Forecasted_AQI": "Forecasted AQI"},
    )
    fig.update_traces(line=dict(color="#168477", width=3), marker=dict(size=6))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        forecast[["Date", "Forecasted_AQI", "Status", "Recommendation"]].rename(
            columns={
                "Date": "Date",
                "Forecasted_AQI": "Forecasted AQI",
                "Status": "Air quality status",
                "Recommendation": "Smart guidance",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Smart recommendation: " + get_recommendation(int(round(float(peak_row['Forecasted_AQI'])))))


def main() -> None:
    st.cache_data.clear()
    df, summary, metrics = get_dashboard_data()
    nav_pages = ["Dashboard", "Trend Analysis", "Smart Alert", "Data Explorer", "Prediction"]

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Dashboard"

    with st.sidebar:
        st.markdown("# 🌿 AeroSense")
        st.caption("AIR QUALITY INTELLIGENCE")
        st.divider()
        sidebar_page = st.radio(
            "NAVIGATION",
            nav_pages,
            index=nav_pages.index(st.session_state.selected_page),
        )
        if sidebar_page != st.session_state.selected_page:
            st.session_state.selected_page = sidebar_page
        st.divider()
        st.caption("Air Quality Analysis & Prediction System")

    page = st.session_state.selected_page
    nav_shell = st.container()
    with nav_shell:
        nav_cols = st.columns(len(nav_pages))
        for column, nav_page in zip(nav_cols, nav_pages):
            with column:
                clicked = st.button(
                    nav_page,
                    key=f"top_nav_{nav_page}",
                    type="primary" if page == nav_page else "secondary",
                    use_container_width=True,
                )
                if clicked:
                    st.session_state.selected_page = nav_page
                    st.rerun()

    page = st.session_state.selected_page

    st.markdown(
        """
        <div class="welcome-banner">
            <div class="title">Air Quality Analysis System</div>
            <div class="subtitle">Monitor historical air pollution data, explore trends and review potential air quality risks.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Dashboard":
        render_dashboard(df, summary)
    elif page == "Trend Analysis":
        render_trend_analysis(df)
    elif page == "Smart Alert":
        render_smart_alert(df)
    elif page == "Data Explorer":
        render_data_explorer(df)
    else:
        render_prediction(df, metrics)



if __name__ == "__main__":
    main()
