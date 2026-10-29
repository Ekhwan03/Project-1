# Air Quality Analysis and Prediction System

This project supports both a desktop application and a website version. It focuses specifically on Malaysia haze conditions for the year 2025, providing historical air quality analysis, machine learning-based forecasting, smart AQI recommendations, and report generation for the approved proposal.

## Features
- Historical air quality analysis with desktop and web interfaces
- AQI trend exploration by location and date
- Pollutant trend views for PM2.5, PM10, NO2, SO2, O3, and CO
- Machine learning forecast for future AQI values
- Smart activity recommendation system based on AQI thresholds and pollution level
- HTML report generation for presentation and review
- SQLite storage for recorded air quality observations

## Tech stack
- Python
- Tkinter (desktop app)
- Streamlit (website app)
- Matplotlib
- Plotly
- Pandas
- scikit-learn
- SQLite

## Run desktop app locally
1. Open a terminal in the project folder.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Launch the desktop app:
   `python app.py`

## Run website app locally
1. Open a terminal in the project folder.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start the website version:
   `python -m streamlit run website_app.py`
4. Open the local URL shown by Streamlit, usually:
   `http://127.0.0.1:8501`

## Data source
The default dataset is a generated 2025 Malaysia haze-focused AQI sample built to demonstrate the project workflow. It can later be replaced with official public datasets such as OpenDOSM or Malaysian environmental monitoring sources restricted to 2025.

## Report generation
The desktop app generates an HTML report and saves it in the project folder as `air_quality_report.html`. The website version can also download a report from the interface.
