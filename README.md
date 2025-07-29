
# Smart Food – Reducing Food Waste in School Canteens

## Overview
**Smart Food** is a data-driven project aimed at reducing food waste in school canteens.  
The system collects, stores, analyzes, and forecasts food consumption and waste data to help schools make better decisions about meal planning.

## Objectives
- Reduce food waste in schools by predicting demand more accurately.
- Provide insights into meal consumption patterns.
- Enable schools to take timely action through dashboards and forecasts.

## Key Features
- **Data Collection and Storage**  
  - School meal and waste data aggregated from multiple Excel files (2023–2025).
  - Stored in **InfluxDB** for time-series management.
- **Data Visualization**  
  - **Grafana** dashboards to explore consumption and waste trends.
- **Forecasting Models**  
  - **Amazon Chronos** for time series forecasting (with fine-tuning).
  - **Pypots** library (TimeMixer + SAITS) for forecasting and missing data imputation.
- **Categorization**  
  - Analysis by school, meal type, macro-category (e.g., pasta, fish, red meat).
  - Configurable categories via a dedicated Excel mapping file.

## Data Analysis
Available types of analysis:
- **Per School** – Waste trends specific to a school.
- **Per Meal** – Waste linked to specific meal types (first course, side dish...).
- **Per Macro-Category** – Aggregated by major food groups.
- **Global** – All schools combined.

### Handling Missing Data
- Missing values are managed with **NaN imputation** in PyPots forecasting.

## Tools and Technologies
- **InfluxDB** – Time-series data storage
- **Grafana** – Dashboards & visualizations
- **Amazon Chronos** – Forecasting (fine-tuned)
- **Pypots** – Forecasting & imputation (TimeMixer, SAITS)
