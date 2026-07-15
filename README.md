# GH EcoDrain-MAP: Strategic NADMO Decision Support System

## Overview

GH EcoDrain-MAP is an intelligent flood risk assessment and decision support system developed for the National Disaster Management Organisation (NADMO) in Ghana. The system combines hydrological modelling, terrain analysis, machine learning, and geospatial data to identify areas vulnerable to flash flooding across Accra.

The application provides an interactive interface for simulating rainfall events, evaluating flood susceptibility, and supporting emergency planning using both rule-based and machine learning approaches.


## Features

- Interactive flood risk simulation
- Machine Learning flood susceptibility prediction (Random Forest)
- Real elevation and slope data (SRTM)
- OpenStreetMap drainage network integration
- Google Earth Engine integration for Sentinel-2 impervious surface analysis
- Interactive Folium map visualization
- Flood hotspot identification
- Critical infrastructure risk assessment
- Decision support dashboard for disaster management

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Folium
- Streamlit-Folium
- Google Earth Engine
- OpenStreetMap (Overpass API)
- OpenTopoData API



## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ecodrainmap.git
cd ecodrainmap
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Google Earth Engine Configuration

To use real Sentinel-2 impervious surface data:

1. Create a Google Cloud Project.
2. Enable the Earth Engine API.
3. Create a Service Account.
4. Download the Service Account JSON key.
5. Configure Streamlit Secrets.

Example:

```toml
gee_project = "your-project-id"
gee_service_account_path = "C:/path/to/service-account.json"
```

---

## Project Structure

```
EcoDrainMap/
│
├── app.py
├── ml_model.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml
└── assets/
```

---

## Data Sources

- Shuttle Radar Topography Mission (SRTM)
- OpenStreetMap
- Google Earth Engine
- Sentinel-2 Imagery
- Open-Meteo Weather API

---

## Machine Learning

The application trains a Random Forest Regressor using a physically derived Flood Susceptibility Index (FSI). The model learns relationships between:

- Elevation
- Slope
- Impervious Surface
- Distance to Drainage
- Rainfall
- Soil Saturation

The trained model predicts flood probability for unseen locations.

---

## Intended Users

- National Disaster Management Organisation (NADMO)
- Metropolitan, Municipal and District Assemblies (MMDAs)
- Urban planners
- Researchers
- Environmental agencies

---

## Author

**Benjamin Abankwah**

BSc Data Science and Analytics

Ghana Communication Technology University

---

## License

This project was developed for academic research and demonstration purposes.