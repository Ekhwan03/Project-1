# Product Requirements Document (PRD)

## 1. Executive Summary
Problem Statement: Many people struggle to understand air quality because raw AQI values and pollutant readings are difficult to interpret without context. The current problem is not just data availability, but the lack of actionable guidance that helps people decide whether outdoor activity is safe and how pollution changes over time.

Proposed Solution: AeroSense is a local air quality analysis and prediction system that helps users explore historical pollution trends, review key pollutants, forecast future AQI values, and receive simple, health-oriented recommendations. The product will combine dashboard visualisation, machine learning forecasting, and practical guidance in one streamlined interface.

Success Criteria:
- The dashboard loads and responds in under 3 seconds for routine navigation between pages.
- AQI prediction model achieves MAE below 15 AQI points on validation data for the test dataset.
- Users can view pollutant trends, AQI classification, and recommendations within 5 seconds of opening a page.
- The system supports at least three primary user groups: students, researchers, and general public users.
- The application runs locally in a low-cost environment with minimal setup and without cloud infrastructure.

## 2. User Experience & Functionality
User Personas:
- Students: Need a project that demonstrates environmental analytics, dashboard design, and machine learning in a practical setting.
- Researchers: Need trend analysis and model evaluation to study pollution behaviour in Malaysian cities.
- General public: Need clear AQI information and actionable recommendations for daily activities and health protection.

User Stories:
- As a student, I want to view historical AQI trends so that I can understand how air quality changes over time.
- As a researcher, I want to compare pollutant trends and location-specific data so that I can evaluate environmental conditions.
- As a public user, I want to see AQI categories and alerts so that I can decide whether outdoor activities are safe.
- As a user, I want a recommendation engine so that I can plan safe activities without needing technical knowledge.
- As a user, I want forecasted AQI values so that I can anticipate air quality conditions for upcoming days.
- As a stakeholder, I want a clean dashboard and downloadable dataset summary so that I can communicate findings clearly to others.

Acceptance Criteria:
- Done: The dashboard displays the latest AQI, average AQI, peak AQI, and key location summaries.
- Done: Users can filter data by location and pollutant to compare historical trends across cities.
- Done: The system categorises AQI into clear health bands such as Good, Moderate, Unhealthy, Very Unhealthy, and Hazardous.
- Done: The prediction page shows a forecast for a selected location and forecast horizon with a visual trend chart.
- Done: Smart alerts provide plain-language recommendations based on AQI thresholds and current conditions.
- Done: Users can search and inspect raw records in the data explorer and download filtered CSV output.
- Done: The application works reliably in a local desktop environment using Streamlit and SQLite-backed data processing.

Non-Goals:
- Real-time streaming sensor ingestion from IoT devices.
- Multi-user authentication and role-based access management.
- Cloud deployment or enterprise monitoring system design.
- Live geospatial map overlays across all regions.
- Regulatory-grade industrial air-quality reporting or compliance workflows.

## 3. AI System Requirements (If Applicable)
Tool Requirements:
- Data processing: Pandas, NumPy, and SQLite for local data management.
- Data visualisation: Plotly for interactive charts and trend exploration.
- Prediction model: scikit-learn RandomForestRegressor for AQI forecasting.
- Front-end: Streamlit for dashboard navigation, controls, and report layout.
- Storage: Local CSV/SQLite dataset with reproducible transformations.
- Output: Dashboard summaries, forecast results, and activity recommendation logic.

Evaluation Strategy:
- Measure prediction quality using MAE and RMSE against a hold-out validation set.
- Compare forecast accuracy across multiple locations and seasonal ranges.
- Validate that alert thresholds align with AQI health categories and recommendation logic.
- Test data filtering and dashboard navigation for usability across common user flows.
- Review output clarity to confirm recommendations are understandable to non-technical users.

## 4. Technical Specifications
Architecture Overview: The system uses a Python-based Streamlit application as the front end. Historical air-quality data is loaded, cleaned, and transformed locally before being processed into summary statistics, charts, and forecast inputs. A machine learning model predicts AQI values based on pollutant readings, meteorological indicators, month, and location metadata. The dashboard then presents trends, recommendations, and alerts in a structured and user-friendly page flow.

Integration Points:
- Data source: historical air-quality dataset for Malaysia, generated or imported as a local CSV/SQLite source.
- Storage: SQLite database and local CSV dataset for data persistence and retrieval.
- Model layer: scikit-learn pipeline for training, validation, and prediction.
- Front-end: Streamlit pages for Dashboard, Trend Analysis, Smart Alert, Data Explorer, and Prediction.
- Reporting: Downloadable filtered dataset and dashboard summary experience for stakeholder review.

Security & Privacy:
- No personal or sensitive user information should be collected or stored.
- All data processing should remain local unless the project is explicitly moved to a secure server environment.
- External dataset sources must be checked for licensing, trustworthiness, and completeness before use in formal deployments.
- The system should avoid exposing internal model details beyond what is necessary for educational or stakeholder demonstration.

## 5. Risks & Roadmap
Phased Rollout:
- MVP: Historical dashboard, pollutant trend analysis, smart alerts, forecast page, and data explorer.
- v1.1: Improved model tuning, refined visual layout, and stronger recommendation logic based on historical AQI thresholds.
- v2.0: Integration with official public datasets, more robust forecasting, and deployment-ready packaging for demonstration or academic use.

Technical Risks:
- Availability and quality of historical data may vary across locations and time periods.
- Forecast accuracy may be weaker during abnormal pollution spikes or unusual seasonal conditions.
- Local deployment may not scale beyond prototype and research use without further infrastructure planning.
- Model performance may differ by region due to data imbalance or limited historical records.

Implementation Guidelines:
- Keep the interface simple, readable, and non-technical for public users.
- Validate AQI thresholds and recommendation logic against real health-risk ranges before promoting them as final guidance.
- Ensure that data cleaning and preprocessing steps are transparent and repeatable.
- Test visual, forecasting, and recommendation flows on realistic scenarios before release.
- Keep the system modular so future dataset sources, forecasting algorithms, and dashboards can be added without major refactors.

Testing and Validation:
- Run automated tests for the prediction pipeline and model output consistency.
- Check dashboard transitions and navigation flows for proper state handling and responsive layout.
- Validate alert recommendations for low, moderate, unhealthy, and hazardous AQI scenarios.
- Confirm that filtered dataset downloads and charts remain accurate when locations or date windows change.
