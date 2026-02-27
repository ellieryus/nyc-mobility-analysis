# NYC Mobility Analysis: Impact of Ride-Hailing on Urban Transportation Patterns

## Project Overview

This project analyzes how ride-hailing platforms (Uber, Lyft) have reshaped urban mobility patterns in New York City over a 16-year period, using NYC Taxi and For-Hire Vehicle (FHV) trip data.

### Research Question

**How have ride-hailing platforms transformed transportation demand, pricing, and geographic service distribution across NYC boroughs?**

### Hypothesis

The growth of ride-hailing platforms significantly reduced traditional taxi trip volumes in Manhattan while increasing overall trip activity in outer boroughs, particularly during off-peak hours.

---

## Project Objectives

1. **Temporal Analysis**: Examine trip volume trends across different time periods (2009-2024)
2. **Spatial Analysis**: Compare geographic distribution of taxi vs ride-hailing services
3. **Demand Patterns**: Analyze peak vs off-peak hour dynamics across boroughs
4. **Pricing Behavior**: Study fare structures and their evolution
5. **Predictive Modeling**: Build models to forecast trip demand patterns

---

## Project Structure

```
nyc-mobility-analysis/
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned, transformed data
│   ├── interim/                # Intermediate processing steps
│   └── external/               # External reference data (borough boundaries, etc.)
│
├── notebooks/
│   ├── exploratory/            # EDA notebooks
│   ├── modeling/               # Model development notebooks
│   └── evaluation/             # Model evaluation and comparison
│
├── src/
│   ├── data/                   # Data acquisition and processing scripts
│   ├── features/               # Feature engineering pipeline
│   ├── models/                 # Model training and prediction
│   ├── visualization/          # Plotting and visualization utilities
│   └── api/                    # API for model serving
│
├── models/                     # Trained model artifacts
├── reports/                    # Generated reports and figures
├── tests/                      # Unit and integration tests
├── config/                     # Configuration files
├── deployment/                 # Deployment configurations
├── docs/                       # Documentation
└── .github/workflows/          # CI/CD pipelines
```

---

## Data Science Lifecycle

This project follows a structured approach:

1. **Business Understanding**
   - Define research questions and hypothesis
   - Identify key stakeholders and use cases

2. **Data Acquisition**
   - Download NYC TLC trip records
   - Acquire supplementary data (weather, events, demographics)

3. **Data Preparation**
   - Clean and validate data
   - Handle missing values and outliers
   - Feature engineering

4. **Exploratory Data Analysis**
   - Temporal trends analysis
   - Spatial distribution patterns
   - Statistical hypothesis testing

5. **Modeling**
   - Time series forecasting (ARIMA, Prophet, LSTM)
   - Demand prediction models
   - Spatial analysis (clustering, hotspot detection)

6. **Evaluation**
   - Model performance metrics
   - Cross-validation strategies
   - A/B testing framework

7. **Deployment**
   - Model serving API
   - Dashboard for insights
   - Automated retraining pipeline

8. **Monitoring**
   - Data drift detection
   - Model performance tracking
   - Alerting system

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda
- Git
- (Optional) Docker for containerization

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nyc-mobility-analysis.git
cd nyc-mobility-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Quick Start

```bash
# 1. Download data
python src/data/download_yellowtaxi_direct.py

# 1b. Build a representative yellow taxi sample (recommended for local dev)
python src/data/sample_yellow_data.py --start-year 2009 --end-year 2025 --sample-size 24000 --output-dir data/samples

# 2. Process data
python src/data/process_data.py

# 3. Run exploratory analysis
jupyter notebook notebooks/exploratory/01_initial_exploration.ipynb

# 4. Train models
python src/models/train_model.py --config config/model_config.yaml

# 5. Generate reports
python src/visualization/generate_report.py
```

Note: if the TLC parquet endpoint is unavailable from your environment, use the
sample script above. It pulls from official NYC Open Data yearly tables and
writes a stratified sample parquet for analysis.

---

## Data Sources

- **Primary**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  - Yellow Taxi trips
  - Green Taxi trips
  - For-Hire Vehicle (FHV) trips
  - High Volume FHV trips (Uber, Lyft)

- **Supplementary** (Optional):
  - NYC Open Data (borough boundaries, census data)
  - Weather data
  - Major events calendar

---

## Technology Stack

- **Data Processing**: pandas, numpy, dask (for large datasets)
- **Visualization**: matplotlib, seaborn, plotly, folium
- **Statistical Analysis**: scipy, statsmodels
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Time Series**: prophet, statsmodels
- **Deep Learning**: TensorFlow/PyTorch (for LSTM models)
- **Geospatial**: geopandas, shapely
- **API**: FastAPI
- **Monitoring**: MLflow, Weights & Biases
- **Testing**: pytest
- **CI/CD**: GitHub Actions

---

## Key Analyses

### 1. Temporal Trends

- Trip volume evolution (2009-2024)
- Seasonal patterns
- Day-of-week and hourly patterns
- Impact of major events (COVID-19, policy changes)

### 2. Spatial Analysis

- Borough-level demand comparison
- Manhattan vs outer boroughs
- Neighborhood hotspot identification
- Service coverage gaps

### 3. Competitive Dynamics

- Market share evolution (taxi vs ride-hailing)
- Price competition analysis
- Service quality metrics

### 4. Predictive Models

- Trip demand forecasting
- Surge pricing prediction
- Optimal fleet allocation

---

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/test_data_processing.py
```

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Data Dictionary](docs/data_dictionary.md)
- [Feature Engineering Guide](docs/feature_engineering.md)
- [Model Documentation](docs/models.md)
- [API Reference](docs/api_reference.md)

---

## Team Members

- Ellie Ha
- Chloe Pham
- Rachelle Dong
- Helia Mohamamdzade
- Maral Vahedi

---

## Acknowledgments

- NYC Taxi and Limousine Commission for providing open data

---

## References

1. NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. [Add relevant research papers]
3. [Add other references]
