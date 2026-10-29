from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox, ttk


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "sample_air_quality_dataset.csv"
DB_PATH = BASE_DIR / "air_quality.db"


def generate_sample_dataset() -> pd.DataFrame:
    location_profiles = {
        "Kuala Lumpur": (138, 62, 38, 75, 24, 1.35),
        "Johor Bahru": (126, 58, 35, 68, 22, 1.18),
        "George Town": (118, 54, 32, 65, 21, 1.08),
        "Kuching": (108, 49, 29, 60, 19, 0.96),
        "Kota Kinabalu": (96, 46, 27, 58, 18, 0.88),
    }
    locations = list(location_profiles.keys())
    records: List[Dict[str, object]] = []
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2025-11-30")

    for location in locations:
        baseline = location_profiles[location]
        for day in pd.date_range(start_date, end_date, freq="D"):
            day_of_year = day.dayofyear
            seasonal = 1 + 0.22 * np.sin((day_of_year / 365) * 2 * np.pi)
            weekday_factor = 1.06 if day.weekday() >= 5 else 1.0
            haze_factor = 1.0 + 0.28 * np.sin((day_of_year / 30.0) + 2.3)
            if 5 <= day.month <= 8:
                haze_factor += 0.25
            loc_factor = 1 + (locations.index(location) * 0.04)

            pm25 = max(18, baseline[0] * seasonal * weekday_factor * loc_factor * haze_factor + np.random.normal(0, 12))
            pm10 = max(25, baseline[1] * seasonal * weekday_factor * loc_factor * haze_factor + np.random.normal(0, 9))
            no2 = max(12, baseline[2] * seasonal * weekday_factor * loc_factor * haze_factor + np.random.normal(0, 6))
            so2 = max(6, baseline[3] * seasonal * weekday_factor * loc_factor * haze_factor + np.random.normal(0, 5))
            o3 = max(14, baseline[4] * seasonal * weekday_factor * loc_factor * haze_factor + np.random.normal(0, 6))
            co = max(0.25, baseline[5] * seasonal * weekday_factor * loc_factor * haze_factor + np.random.normal(0, 0.18))
            temperature = 29 + 7 * np.sin((day_of_year / 365) * 2 * np.pi - 1.2) + np.random.normal(0, 2.6)
            humidity = 72 + 20 * np.cos((day_of_year / 365) * 2 * np.pi - 0.6) + np.random.normal(0, 8)
            rainfall = max(0, 30 + 18 * np.sin((day_of_year / 365) * 2 * np.pi + 1.2) + np.random.normal(0, 15))

            aqi = 0.45 * pm25 + 0.22 * pm10 + 0.16 * no2 + 0.08 * so2 + 0.09 * o3 + 0.10 * co * 10
            aqi += 18 * np.sin(day_of_year / 22.0) + np.random.normal(0, 10)
            aqi = max(40, min(300, round(aqi)))

            records.append(
                {
                    "Date": day.strftime("%Y-%m-%d"),
                    "Location": location,
                    "PM2.5": round(pm25, 2),
                    "PM10": round(pm10, 2),
                    "NO2": round(no2, 2),
                    "SO2": round(so2, 2),
                    "O3": round(o3, 2),
                    "CO": round(co, 2),
                    "Temperature_C": round(temperature, 2),
                    "Humidity_%": round(humidity, 2),
                    "Rainfall_mm": round(rainfall, 2),
                    "AQI": int(aqi),
                }
            )

    df = pd.DataFrame(records)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    return df


def load_dataset(selected_year: str | int | None = None) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if not DATASET_PATH.exists():
        generate_sample_dataset()
    else:
        current_df = pd.read_csv(DATASET_PATH)
        if not current_df.empty and "Date" in current_df.columns:
            current_dates = pd.to_datetime(current_df["Date"], errors="coerce")
            if current_dates.notna().any():
                min_date = current_dates.min()
                max_date = current_dates.max()
                if min_date > pd.Timestamp("2023-01-01") or max_date < pd.Timestamp("2025-11-30"):
                    generate_sample_dataset()

    df = pd.read_csv(DATASET_PATH)
    required = ["Date", "Location", "PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "Temperature_C", "Humidity_%", "Rainfall_mm", "AQI"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2025-11-30")].reset_index(drop=True)

    if selected_year is not None and str(selected_year).lower() != "all":
        selected_year = int(selected_year)
        df = df[df["Date"].dt.year == selected_year].reset_index(drop=True)

    if df.empty:
        if selected_year is not None and str(selected_year).lower() != "all":
            raise ValueError(f"No valid records are available for year {selected_year}.")
        raise ValueError("No valid 2023-2025 haze records are available in the dataset.")

    return df.sort_values("Date").reset_index(drop=True)


def get_aqi_summary() -> Dict[str, object]:
    df = load_dataset()
    summary = {
        "records": len(df),
        "avg_aqi": round(float(df["AQI"].mean()), 2),
        "max_aqi": int(df["AQI"].max()),
        "min_aqi": int(df["AQI"].min()),
        "locations": sorted(df["Location"].unique().tolist()),
        "latest_date": df["Date"].max().strftime("%Y-%m-%d"),
        "peak_location": df.loc[df["AQI"].idxmax(), "Location"],
    }
    return summary


def classify_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_recommendation(aqi: float) -> str:
    if aqi <= 50:
        return "Good air quality. Recommended activities: jogging, cycling, outdoor sports, family picnic, and brisk walking."
    if aqi <= 100:
        return "Moderate air quality. Recommended activities: light walking, short outdoor exercise, and early morning activities. Avoid long sessions for sensitive groups."
    if aqi <= 150:
        return "Unhealthy for sensitive groups. Recommended activities: indoor workouts, stretching, light home exercise, and shorter outdoor exposure."
    if aqi <= 200:
        return "Unhealthy air quality. Recommended activities: indoor gym, yoga, meditation, and desk-based activities. Avoid strenuous outdoor exercise."
    if aqi <= 300:
        return "Very unhealthy air quality. Recommended activities: indoor relaxation, reading, indoor games, and light mobility exercises at home. Stay indoors."
    return "Hazardous air quality. Recommended activities: stay indoors only, practise light indoor mobility, and avoid all outdoor exertion."


def get_forecast_summary(forecast: pd.DataFrame, location: str) -> str:
    if forecast.empty:
        return f"No 2023-2025 haze forecast is available for {location}."

    avg_aqi = float(forecast["Forecasted_AQI"].mean())
    peak_row = forecast.loc[forecast["Forecasted_AQI"].idxmax()]
    peak_status = classify_aqi(float(peak_row["Forecasted_AQI"]))
    highest_day = pd.to_datetime(peak_row["Date"]).strftime("%b %d, %Y")

    if avg_aqi <= 50:
        outlook = "clearer conditions with light haze impact"
    elif avg_aqi <= 100:
        outlook = "moderate haze but still manageable for short outdoor trips"
    elif avg_aqi <= 150:
        outlook = "noticeable haze and higher risk for sensitive groups"
    elif avg_aqi <= 200:
        outlook = "heavy haze conditions that call for indoor activities"
    elif avg_aqi <= 300:
        outlook = "very unhealthy haze conditions requiring serious caution"
    else:
        outlook = "hazardous haze conditions and should be treated as stay-inside weather"

    return (
        f"2023-2025 haze outlook for {location}: average forecast AQI is {avg_aqi:.1f}, with the heaviest haze reaching "
        f"{peak_row['Forecasted_AQI']:.1f} AQI on {highest_day} ({peak_status}). The next {len(forecast)} days are expected to show {outlook}."
    )


def train_prediction_model() -> Dict[str, object]:
    df = load_dataset().copy()
    df["Month"] = df["Date"].dt.month
    df["Location_Code"] = df["Location"].astype("category").cat.codes

    feature_cols = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "O3",
        "CO",
        "Temperature_C",
        "Humidity_%",
        "Rainfall_mm",
        "Month",
        "Location_Code",
    ]

    X = df[feature_cols]
    y = df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=250, random_state=42, max_depth=15)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = model.score(X_test, y_test)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def forecast_future_aqi(location: str, days: int = 7, selected_year: str | int | None = None) -> pd.DataFrame:
    df = load_dataset()
    if selected_year is not None and str(selected_year).lower() != "all":
        df = df[df["Date"].dt.year == int(selected_year)].reset_index(drop=True)

    loc_df = df[df["Location"] == location].sort_values("Date").copy()
    if loc_df.empty:
        return pd.DataFrame(columns=["Date", "Forecasted_AQI", "Status", "Forecast_Label", "Recommendation"])

    model_data = train_prediction_model()
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    location_code = df["Location"].astype("category").cat.codes.loc[df["Location"] == location].iloc[0]
    last_row = loc_df.iloc[-1].copy()
    last_date = pd.Timestamp(last_row["Date"])
    upper_bound = pd.Timestamp("2025-12-31")
    max_allowed_days = max(0, (upper_bound - last_date).days)
    effective_days = min(days, max_allowed_days)
    forecast_rows = []

    for i in range(1, effective_days + 1):
        future_date = last_date + pd.Timedelta(days=i)

        seasonal_wave = 0.92 + 0.18 * np.sin((future_date.dayofyear / 16.5) + 1.2)
        if future_date.month in [5, 6, 7, 8, 9]:
            seasonal_wave += 0.12
        if future_date.month in [1, 2, 11, 12]:
            seasonal_wave += 0.06

        oscillation = 1.0 + 0.16 * np.sin(i / 2.7)
        dampening = 1.0 - (0.035 * i / max(1, effective_days))
        location_bias = 1.0 + (location_code * 0.035)

        baseline_pm25 = last_row["PM2.5"] * seasonal_wave * oscillation * dampening * location_bias
        baseline_pm10 = last_row["PM10"] * seasonal_wave * (1.0 + 0.05 * np.sin(i / 3.4)) * dampening * location_bias
        baseline_no2 = last_row["NO2"] * seasonal_wave * (1.0 + 0.06 * np.cos(i / 3.0)) * dampening
        baseline_so2 = last_row["SO2"] * seasonal_wave * (1.0 + 0.08 * np.sin(i / 2.5)) * dampening
        baseline_o3 = last_row["O3"] * seasonal_wave * (1.0 + 0.04 * np.sin(i / 2.1)) * dampening
        baseline_co = last_row["CO"] * (1.0 + 0.10 * np.sin(i / 2.8)) * (0.94 + 0.08 * np.cos(i / 3.3))

        feature_values = {
            "PM2.5": max(18, baseline_pm25),
            "PM10": max(24, baseline_pm10),
            "NO2": max(10, baseline_no2),
            "SO2": max(7, baseline_so2),
            "O3": max(13, baseline_o3),
            "CO": max(0.22, baseline_co),
            "Temperature_C": last_row["Temperature_C"] + 1.2 * np.sin(i / 1.9) - 0.3,
            "Humidity_%": min(100, max(35, last_row["Humidity_%"] + 2.1 * np.cos(i / 2.3) - 0.8)),
            "Rainfall_mm": max(0, last_row["Rainfall_mm"] + 4.0 * np.sin(i / 2.7) - 1.8),
            "Month": future_date.month,
            "Location_Code": location_code,
        }
        feature_df = pd.DataFrame([feature_values])[feature_cols]
        prediction = float(model.predict(feature_df)[0])
        prediction *= 0.82 + 0.34 * np.sin((i * 1.7) + 0.9)
        prediction *= 1.0 - (0.03 * i / max(1, effective_days))
        prediction = max(40.0, min(220.0, prediction))
        status = classify_aqi(prediction)
        recommendation = get_recommendation(prediction)
        forecast_rows.append(
            {
                "Date": future_date.strftime("%Y-%m-%d"),
                "Forecasted_AQI": round(prediction, 2),
                "Status": status,
                "Forecast_Label": f"2023-2025 haze forecast: {status}",
                "Recommendation": recommendation,
            }
        )

        last_row = last_row.copy()
        last_row["PM2.5"] = feature_values["PM2.5"]
        last_row["PM10"] = feature_values["PM10"]
        last_row["NO2"] = feature_values["NO2"]
        last_row["SO2"] = feature_values["SO2"]
        last_row["O3"] = feature_values["O3"]
        last_row["CO"] = feature_values["CO"]
        last_row["Temperature_C"] = feature_values["Temperature_C"]
        last_row["Humidity_%"] = feature_values["Humidity_%"]
        last_row["Rainfall_mm"] = feature_values["Rainfall_mm"]

    return pd.DataFrame(forecast_rows)


def initialise_database() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS air_quality_records (Date TEXT, Location TEXT, PM2_5 REAL, PM10 REAL, NO2 REAL, SO2 REAL, O3 REAL, CO REAL, Temperature_C REAL, Humidity REAL, Rainfall_mm REAL, AQI INTEGER)"
    )
    df = load_dataset().rename(columns={"Humidity_%": "Humidity"})
    existing_rows = pd.read_sql_query("SELECT COUNT(*) AS total FROM air_quality_records", conn).iloc[0]["total"]
    if existing_rows == 0:
        df[["Date", "Location", "PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "Temperature_C", "Humidity", "Rainfall_mm", "AQI"]].to_sql(
            "air_quality_records",
            conn,
            if_exists="append",
            index=False,
        )
    conn.close()


def build_report_html(df: pd.DataFrame, metrics: Dict[str, float], forecast: pd.DataFrame) -> str:
    summary = df.groupby("Location")["AQI"].mean().sort_values(ascending=False).reset_index()
    summary.columns = ["Location", "Average AQI"]
    latest = df.sort_values("Date").tail(1).iloc[0]

    table_rows = "".join(
        f'<tr><td>{row["Location"]}</td><td>{row["Average AQI"]:.2f}</td></tr>'
        for _, row in summary.iterrows()
    )
    forecast_rows = "".join(
        f'<tr><td>{row["Date"]}</td><td>{row["Forecasted_AQI"]}</td><td>{row["Status"]}</td></tr>'
        for _, row in forecast.iterrows()
    )

    return f"""
    <html>
      <head><title>2023-2025 Malaysia Haze Analysis Report</title></head>
      <body>
        <h1>Air Quality Analysis and Prediction System</h1>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h2>Overview</h2>
        <ul>
          <li>Average AQI: {df['AQI'].mean():.2f}</li>
          <li>Highest AQI: {df['AQI'].max()}</li>
          <li>Latest Recorded Date: {latest['Date']}</li>
          <li>Latest Location: {latest['Location']}</li>
        </ul>
        <h2>Model Performance</h2>
        <ul>
          <li>MAE: {metrics.get('mae', 0):.2f}</li>
          <li>RMSE: {metrics.get('rmse', 0):.2f}</li>
          <li>R²: {metrics.get('r2', 0):.3f}</li>
        </ul>
        <h2>Average AQI by Location</h2>
        <table>
          <tr><th>Location</th><th>Average AQI</th></tr>
          {table_rows}
        </table>
        <h2>2023-2025 Haze Forecast</h2>
        <table>
          <tr><th>Date</th><th>Forecasted AQI</th><th>Status</th></tr>
          {forecast_rows}
        </table>
      </body>
    </html>
    """


def create_chart(parent: tk.Misc, title: str, x_values, y_values, ylabel: str) -> FigureCanvasTkAgg:
    fig = Figure(figsize=(6, 3.4), dpi=100)
    plot = fig.add_subplot(111)
    plot.plot(x_values, y_values, marker="o", linewidth=2.2, color="#168477")
    plot.set_title(title, color="#123b37", fontsize=11, fontweight="bold")
    plot.set_xlabel("Date", color="#123b37")
    plot.set_ylabel(ylabel, color="#123b37")
    plot.tick_params(axis="x", colors="#123b37")
    plot.tick_params(axis="y", colors="#123b37")
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas


def create_multi_series_chart(parent: tk.Misc, title: str, series: List[tuple[str, list, list]], ylabel: str) -> FigureCanvasTkAgg:
    fig = Figure(figsize=(6, 3.4), dpi=100)
    plot = fig.add_subplot(111)
    colors = ["#168477", "#2c7fb8", "#d97706"]

    for index, (label, x_values, y_values) in enumerate(series):
        plot.plot(x_values, y_values, marker="o", linewidth=2.0, color=colors[index % len(colors)], label=label)

    plot.set_title(title, color="#123b37", fontsize=11, fontweight="bold")
    plot.set_xlabel("Date", color="#123b37")
    plot.set_ylabel(ylabel, color="#123b37")
    plot.tick_params(axis="x", colors="#123b37")
    plot.tick_params(axis="y", colors="#123b37")
    plot.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas


class ScrollableFrame(tk.Frame):
    def __init__(self, parent: tk.Misc, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=self["bg"] if "bg" in kwargs else "#f3faf8")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=self["bg"] if "bg" in kwargs else "#f3faf8")
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.bind("<Configure>", self._on_parent_configure)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag_scroll)

        self._drag_start_y = 0
        self._drag_start_scroll = 0

    def _on_mousewheel(self, event: tk.Event) -> None:
        if hasattr(event, "delta") and event.delta:
            delta = int(-event.delta / 120)
            self.canvas.yview_scroll(delta, "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_start_y = event.y
        self._drag_start_scroll = self.canvas.yview()[0]

    def _drag_scroll(self, event: tk.Event) -> None:
        delta = (event.y - self._drag_start_y) / 2
        self.canvas.yview_moveto(self._drag_start_scroll - delta / max(1, self.canvas.winfo_height()))

    def _on_inner_configure(self, _event: object) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_parent_configure(self, event: object) -> None:
        width = max(200, event.width - self.scrollbar.winfo_reqwidth())
        self.canvas.configure(width=width)


class AQIApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AeroSense | Air Quality System")
        self.root.geometry("420x860")
        self.root.minsize(380, 680)
        self.root.configure(bg="#081d2d")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("AppHeader.TFrame", background="#0b1f2d")
        style.configure("Custom.TNotebook", background="#081d2d", borderwidth=0)
        style.configure("Custom.TNotebook.Tab", background="#1b3242", foreground="#ecfdf7", padding=(18, 10), borderwidth=0)
        style.map("Custom.TNotebook.Tab", background=[("selected", "#1db8a8"), ("active", "#2a4e5a")], foreground=[("selected", "#ffffff"), ("active", "#ecfdf7")])

        header = tk.Frame(self.root, bg="#0b1f2d", height=110)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        header.configure(padx=6, pady=0)

        brand = tk.Label(
            header,
            text="🌿 AeroSense",
            font=("Segoe UI", 24, "bold"),
            bg="#0b1f2d",
            fg="#f4fffd",
            anchor="w",
        )
        brand.place(x=28, y=28)

        subheading = tk.Label(
            header,
            text="Air Quality Intelligence",
            font=("Segoe UI", 10, "bold"),
            bg="#0b1f2d",
            fg="#bfe8df",
            anchor="w",
        )
        subheading.place(x=32, y=68)

        initialise_database()
        self.df = load_dataset()

        self.notebook = ttk.Notebook(self.root, style="Custom.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))

        self.overview_tab = tk.Frame(self.notebook, bg="#f3faf8")
        self.prediction_tab = tk.Frame(self.notebook, bg="#f3faf8")
        self.alerts_tab = tk.Frame(self.notebook, bg="#f3faf8")
        self.report_tab = tk.Frame(self.notebook, bg="#f3faf8")

        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.prediction_tab, text="Prediction")
        self.notebook.add(self.alerts_tab, text="Smart Alerts")
        self.notebook.add(self.report_tab, text="Reports")

        self.build_overview_tab()
        self.build_prediction_tab()
        self.build_alerts_tab()
        self.build_reports_tab()

    def build_overview_tab(self) -> None:
        self.overview_tab.configure(bg="#f3faf8")
        summary_frame = tk.Frame(self.overview_tab, bg="#f3faf8")
        summary_frame.pack(fill=tk.X, padx=16, pady=(16, 0))

        summary = get_aqi_summary()
        cards = [
            ("Records", summary["records"]),
            ("Average AQI", f"{summary['avg_aqi']:.2f}"),
            ("Peak AQI", summary["max_aqi"]),
            ("Peak Location", summary["peak_location"]),
        ]
        for idx, (label, val) in enumerate(cards):
            card = tk.Label(
                summary_frame,
                text=f"{label}\n{val}",
                font=("Segoe UI", 12, "bold"),
                bg="#edfaf7",
                fg="#123b37",
                relief="solid",
                bd=1,
                padx=10,
                pady=12,
                width=17,
                borderwidth=1,
            )
            card.grid(row=0, column=idx, padx=10, pady=10, sticky="nsew")

        controls = tk.Frame(self.overview_tab, bg="#f3faf8")
        controls.pack(fill=tk.X, padx=16, pady=14)

        location_label = tk.Label(controls, text="Location:", bg="#f3faf8", fg="#123b37", font=("Segoe UI", 10, "bold"))
        location_label.grid(row=0, column=0, padx=(0, 10), sticky="w")
        self.location_var = tk.StringVar(value="All")
        locations = ["All"] + summary["locations"]
        self.location_combo = ttk.Combobox(controls, textvariable=self.location_var, values=locations, state="readonly", width=26)
        self.location_combo.grid(row=0, column=1, sticky="w")
        self.location_combo.bind("<<ComboboxSelected>>", self.render_overview_charts)

        year_label = tk.Label(controls, text="Year:", bg="#f3faf8", fg="#123b37", font=("Segoe UI", 10, "bold"))
        year_label.grid(row=0, column=2, padx=(20, 10), sticky="w")
        years = ["All"] + [str(year) for year in sorted(load_dataset()["Date"].dt.year.unique().tolist())]
        self.year_var = tk.StringVar(value="All")
        self.year_combo = ttk.Combobox(controls, textvariable=self.year_var, values=years, state="readonly", width=12)
        self.year_combo.grid(row=0, column=3, sticky="w")
        self.year_combo.bind("<<ComboboxSelected>>", self.render_overview_charts)

        pollutant_label = tk.Label(controls, text="Pollutant:", bg="#f3faf8", fg="#123b37", font=("Segoe UI", 10, "bold"))
        pollutant_label.grid(row=0, column=4, padx=(20, 10), sticky="w")
        self.pollutant_var = tk.StringVar(value="PM2.5")
        self.pollutant_combo = ttk.Combobox(controls, textvariable=self.pollutant_var, values=["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"], state="readonly", width=18)
        self.pollutant_combo.grid(row=0, column=5, sticky="w")
        self.pollutant_combo.bind("<<ComboboxSelected>>", self.render_overview_charts)

        self.chart_container = ScrollableFrame(self.overview_tab, bg="#f3faf8")
        self.chart_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        self.render_overview_charts()

    def render_overview_charts(self, event=None) -> None:
        for widget in self.chart_container.inner.winfo_children():
            widget.destroy()

        df = load_dataset()
        selected = self.location_var.get()
        selected_year = self.year_var.get()
        if selected_year != "All":
            df = df[df["Date"].dt.year == int(selected_year)].reset_index(drop=True)
        filtered = df if selected == "All" else df[df["Location"] == selected]

        trend_frame = tk.Frame(self.chart_container.inner, bg="#f3f6fb")
        trend_frame.pack(fill=tk.BOTH, expand=True)
        if selected_year == "All":
            yearly_series = []
            for year in sorted(filtered["Date"].dt.year.unique().tolist()):
                year_df = filtered[filtered["Date"].dt.year == year].groupby("Date")["AQI"].mean().reset_index()
                yearly_series.append((str(year), year_df["Date"].tolist(), year_df["AQI"].tolist()))
            create_multi_series_chart(trend_frame, "Yearly AQI Trend", yearly_series, "AQI")
        else:
            chart_df = filtered.groupby("Date")["AQI"].mean().reset_index()
            create_chart(trend_frame, "Daily AQI Trend", chart_df["Date"], chart_df["AQI"], "AQI")

        pollutant_choice = self.pollutant_var.get()
        poll_frame = tk.Frame(self.chart_container.inner, bg="#f3f6fb")
        poll_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        if selected_year == "All":
            pollutant_series = []
            for year in sorted(filtered["Date"].dt.year.unique().tolist()):
                year_df = filtered[filtered["Date"].dt.year == year].groupby("Date")[pollutant_choice].mean().reset_index()
                pollutant_series.append((str(year), year_df["Date"].tolist(), year_df[pollutant_choice].tolist()))
            create_multi_series_chart(poll_frame, f"{pollutant_choice} Trend by Year", pollutant_series, pollutant_choice)
        else:
            pollutant_df = filtered.groupby("Date")[pollutant_choice].mean().reset_index()
            create_chart(poll_frame, f"{pollutant_choice} Trend", pollutant_df["Date"], pollutant_df[pollutant_choice], pollutant_choice)

        summary_text = tk.Text(self.chart_container.inner, height=14, width=86, font=("Segoe UI", 10), wrap="word")
        summary_text.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        location_summary = filtered.groupby("Location")["AQI"].agg(["mean", "min", "max"]).reset_index()
        location_summary.columns = ["Location", "Average AQI", "Minimum AQI", "Maximum AQI"]
        summary_text.insert(tk.END, "Location summary\n" + location_summary.to_string(index=False))
        summary_text.config(state=tk.DISABLED)

    def build_prediction_tab(self) -> None:
        self.prediction_tab.configure(bg="#f3faf8")
        tk.Label(self.prediction_tab, text="2023-2025 Haze Forecast", font=("Segoe UI", 18, "bold"), bg="#f3faf8", fg="#123b37").pack(anchor="w", padx=16, pady=(16, 8))

        controls = tk.Frame(self.prediction_tab, bg="#f3faf8")
        controls.pack(fill=tk.X, padx=16, pady=10)

        location_label = tk.Label(controls, text="Location:", font=("Segoe UI", 10, "bold"), bg="#f3faf8", fg="#123b37")
        location_label.grid(row=0, column=0, padx=(0, 10), sticky="w")
        locations = sorted(load_dataset()["Location"].unique().tolist())
        self.prediction_location_var = tk.StringVar(value=locations[0])
        self.prediction_location_combo = ttk.Combobox(controls, textvariable=self.prediction_location_var, values=locations, state="readonly", width=24)
        self.prediction_location_combo.grid(row=0, column=1, sticky="w")

        year_label = tk.Label(controls, text="Year:", font=("Segoe UI", 10, "bold"), bg="#f3faf8", fg="#123b37")
        year_label.grid(row=0, column=2, padx=(20, 10), sticky="w")
        years = [str(year) for year in sorted(load_dataset()["Date"].dt.year.unique().tolist())]
        self.prediction_year_var = tk.StringVar(value=years[-1])
        self.prediction_year_combo = ttk.Combobox(controls, textvariable=self.prediction_year_var, values=years, state="readonly", width=10)
        self.prediction_year_combo.grid(row=0, column=3, sticky="w")
        self.prediction_year_combo.bind("<<ComboboxSelected>>", lambda event: self.render_prediction_data())

        days_label = tk.Label(controls, text="Forecast days:", font=("Segoe UI", 10, "bold"), bg="#f3faf8", fg="#123b37")
        days_label.grid(row=0, column=4, padx=(20, 10), sticky="w")
        self.forecast_days_var = tk.IntVar(value=7)
        tk.Spinbox(controls, from_=3, to=30, textvariable=self.forecast_days_var, width=8).grid(row=0, column=5, sticky="w")

        update_btn = tk.Button(controls, text="Generate Forecast", command=self.render_prediction_data, bg="#1db8a8", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=6)
        update_btn.grid(row=0, column=6, padx=(20, 0), sticky="w")

        self.metric_frame = tk.Frame(self.prediction_tab, bg="#f3faf8")
        self.metric_frame.pack(fill=tk.X, padx=16, pady=(0, 10))

        self.summary_label = tk.Label(self.prediction_tab, text="", justify=tk.LEFT, wraplength=1000, font=("Segoe UI", 11), bg="#f3faf8", fg="#234d7a")
        self.summary_label.pack(fill=tk.X, padx=16, pady=(0, 8))

        self.prediction_chart_frame = ScrollableFrame(self.prediction_tab, bg="#f3faf8")
        self.prediction_chart_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        self.render_prediction_data()

    def render_prediction_data(self) -> None:
        for widget in self.metric_frame.winfo_children():
            widget.destroy()
        for widget in self.prediction_chart_frame.inner.winfo_children():
            widget.destroy()

        model_data = train_prediction_model()
        location = self.prediction_location_var.get()
        days = int(self.forecast_days_var.get())

        metrics = [
            ("Model MAE", f"{model_data['mae']:.2f}"),
            ("Model RMSE", f"{model_data['rmse']:.2f}"),
            ("Model R²", f"{model_data['r2']:.3f}"),
        ]
        for idx, (label, value) in enumerate(metrics):
            metric = tk.Label(self.metric_frame, text=f"{label}: {value}", font=("Segoe UI", 11, "bold"), bg="#eaf2ff", relief="solid", padx=12, pady=10)
            metric.grid(row=0, column=idx, padx=8, pady=4, sticky="nsew")

        selected_year = self.prediction_year_var.get()
        forecast = forecast_future_aqi(location, days=days, selected_year=selected_year)
        self.summary_label.config(text=get_forecast_summary(forecast, location))

        chart_panel = tk.Frame(self.prediction_chart_frame.inner, bg="#f3f6fb")
        chart_panel.pack(fill=tk.BOTH, expand=True)
        create_chart(chart_panel, f"2023-2025 haze forecast for {location}", forecast["Date"], forecast["Forecasted_AQI"], "Forecast AQI")

        tree = ttk.Treeview(self.prediction_chart_frame.inner, columns=("Date", "Forecasted_AQI", "Status", "Recommendation"), show="headings", height=10)
        tree.heading("Date", text="Date")
        tree.heading("Forecasted_AQI", text="Forecasted AQI")
        tree.heading("Status", text="Status")
        tree.heading("Recommendation", text="Smart guidance")
        tree.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        for _, row in forecast.iterrows():
            tree.insert("", tk.END, values=(row["Date"], row["Forecasted_AQI"], row["Status"], row["Recommendation"]))

    def build_alerts_tab(self) -> None:
        self.alerts_tab.configure(bg="#f3faf8")
        df = load_dataset()
        latest = df.sort_values("Date").tail(1).iloc[0]
        current_aqi = int(latest["AQI"])

        tk.Label(self.alerts_tab, text="Smart Activity Recommendation System", font=("Segoe UI", 16, "bold"), bg="#f3faf8", fg="#123b37").pack(anchor="w", padx=20, pady=(20, 10))

        status_frame = tk.Frame(self.alerts_tab, bg="#f3faf8")
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        status_text = f"{current_aqi} ({classify_aqi(current_aqi)})"
        tk.Label(status_frame, text=status_text, font=("Segoe UI", 16, "bold"), fg="#1849a9", bg="#eaf2ff", padx=12, pady=14, relief="solid").pack(anchor="w")

        recommendation_box = tk.Text(self.alerts_tab, height=8, wrap=tk.WORD, font=("Segoe UI", 11), bg="#ffffff", fg="#123b37")
        recommendation_box.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        recommendation_box.insert(tk.END, get_recommendation(current_aqi))
        recommendation_box.config(state=tk.DISABLED)

        controls = tk.Frame(self.alerts_tab, bg="#f3faf8")
        controls.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(controls, text="Manual AQI check:", font=("Segoe UI", 10, "bold"), bg="#f3faf8", fg="#123b37").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.manual_aqi_var = tk.IntVar(value=current_aqi)
        tk.Entry(controls, textvariable=self.manual_aqi_var, width=12, font=("Segoe UI", 11)).grid(row=0, column=1, sticky="w")
        tk.Button(controls, text="Check Status", command=self.update_manual_alert, bg="#1db8a8", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, pady=6).grid(row=0, column=2, padx=(20, 0), sticky="w")

        self.manual_result_var = tk.StringVar()
        tk.Label(self.alerts_tab, textvariable=self.manual_result_var, font=("Segoe UI", 12, "bold"), fg="#1f5d2f", bg="#f3faf8", padx=10, pady=8).pack(anchor="w", padx=20, pady=(5, 10))
        self.update_manual_alert()

    def update_manual_alert(self) -> None:
        aqi = float(self.manual_aqi_var.get())
        status = classify_aqi(aqi)
        text = f"Classification: {status} | Recommendation: {get_recommendation(aqi)}"
        self.manual_result_var.set(text)

    def build_reports_tab(self) -> None:
        self.report_tab.configure(bg="#f3faf8")
        tk.Label(self.report_tab, text="Generate analytical report", font=("Segoe UI", 16, "bold"), bg="#f3faf8", fg="#123b37").pack(anchor="w", padx=20, pady=(20, 10))

        button_frame = tk.Frame(self.report_tab, bg="#f3faf8")
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Button(button_frame, text="Generate Report", command=self.generate_report, bg="#1db8a8", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, pady=6).pack(anchor="w")

        self.report_text = tk.Text(self.report_tab, height=26, wrap=tk.WORD, font=("Segoe UI", 10), bg="#ffffff", fg="#123b37")
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        self.generate_report()

    def generate_report(self) -> None:
        df = load_dataset()
        metrics = train_prediction_model()
        location = df["Location"].mode().iloc[0]
        forecast = forecast_future_aqi(location, days=7)
        html_report = build_report_html(df, metrics, forecast)

        summary = "2023-2025 Malaysia Haze Analysis and Prediction System\n\n"
        summary += f"Records: {len(df)}\n"
        summary += f"Average AQI: {df['AQI'].mean():.2f}\n"
        summary += f"Peak AQI: {df['AQI'].max()}\n"
        summary += f"Model MAE: {metrics['mae']:.2f}\n"
        summary += f"Model RMSE: {metrics['rmse']:.2f}\n"
        summary += f"Model R²: {metrics['r2']:.3f}\n\n"
        summary += "Forecast preview:\n"
        summary += forecast.to_string(index=False)

        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, summary)

        report_path = BASE_DIR / "air_quality_report.html"
        report_path.write_text(html_report, encoding="utf-8")
        messagebox.showinfo("Report Ready", f"Saved report to:\n{report_path}")


def main() -> None:
    root = tk.Tk()
    AQIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

