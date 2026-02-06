# NYC Mobility Analysis - Project Summary

## Quick Start Guide

### 1. Initial Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/nyc-mobility-analysis.git
cd nyc-mobility-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev
# OR manually:
pip install -r requirements.txt
pip install -e .
```

### 2. Download Data (varies by date range)

```bash
# Download last 2 years of data
make data-download
# OR manually:
python src/data/download_data.py --start-year 2022 --end-year 2024
```

### 3. Process Data (10-30 minutes)

```bash
# Process raw data
make data-process
# OR manually:
python src/data/process_data.py
```

### 4. Explore Data (interactive)

```bash
# Start Jupyter notebook
make notebook
# OR manually:
jupyter notebook notebooks/

# Open: notebooks/exploratory/01_initial_exploration.ipynb
```

### 5. Run Analysis

```bash
# Run tests
make test

# Start API server
make serve

# Access API documentation: http://localhost:8000/docs
```

---

## Project Structure Overview

```
nyc-mobility-analysis/
│
├── README.md              # Main project documentation
├── setup.py               # Package installation configuration
├── requirements.txt       # Python dependencies
├── Makefile              # Common commands
├── .env.template         # Environment variables template
├── docker-compose.yml    # Docker setup
│
├── data/                 # Data directory (not in git)
│   ├── raw/             # Downloaded raw data
│   ├── processed/       # Cleaned data
│   ├── interim/         # Intermediate processing
│   └── external/        # External reference data
│
├── notebooks/           # Jupyter notebooks
│   ├── exploratory/    # EDA notebooks
│   ├── modeling/       # Model development
│   └── evaluation/     # Model evaluation
│
├── src/                # Source code
│   ├── data/          # Data processing
│   │   ├── download_data.py
│   │   └── process_data.py
│   ├── features/      # Feature engineering
│   ├── models/        # Model training
│   ├── visualization/ # Plotting utilities
│   └── api/          # FastAPI application
│       └── app.py
│
├── tests/             # Unit tests
├── models/            # Trained models (not in git)
├── reports/           # Generated reports
├── config/            # Configuration files
├── deployment/        # Deployment configs
└── docs/             # Documentation
    ├── data_dictionary.md
    └── ...
```

---

## Data Science Lifecycle Phases

### Phase 1: Business Understanding ✅
- **Completed**: Research question and hypothesis defined
- **Next**: Validate with stakeholders

### Phase 2: Data Acquisition
- **Status**: In Progress
- **Tasks**:
  - [ ] Download historical data (2009-2024)
  - [ ] Acquire supplementary data (weather, events)
  - [ ] Document data sources

### Phase 3: Data Preparation
- **Status**: In Progress
- **Tasks**:
  - [ ] Clean and validate data
  - [ ] Handle missing values
  - [ ] Create standardized schema
  - [ ] Engineer temporal features
  - [ ] Engineer spatial features

### Phase 4: Exploratory Analysis
- **Status**: Ready to Start
- **Tasks**:
  - [ ] Temporal trend analysis
  - [ ] Spatial distribution analysis
  - [ ] Statistical hypothesis testing
  - [ ] Identify patterns and anomalies

### Phase 5: Modeling
- **Status**: Not Started
- **Planned Models**:
  - [ ] Time series forecasting (ARIMA, Prophet, LSTM)
  - [ ] Regression models (XGBoost, LightGBM)
  - [ ] Clustering for spatial analysis

### Phase 6: Evaluation
- **Status**: Not Started
- **Tasks**:
  - [ ] Cross-validation
  - [ ] Metrics calculation
  - [ ] Model comparison
  - [ ] Hypothesis testing

### Phase 7: Deployment
- **Status**: Infrastructure Ready
- **Tasks**:
  - [ ] Containerize application
  - [ ] Deploy API
  - [ ] Create dashboard
  - [ ] Setup monitoring

### Phase 8: Monitoring
- **Status**: Not Started
- **Tasks**:
  - [ ] Data drift detection
  - [ ] Model performance tracking
  - [ ] Automated retraining

---

## Key Files and Their Purpose

### Configuration Files
- `config/model_config.yaml` - Model hyperparameters and settings
- `.env.template` - Environment variables template
- `setup.py` - Package installation configuration

### Data Processing
- `src/data/download_data.py` - Downloads NYC TLC data
- `src/data/process_data.py` - Cleans and processes raw data

### Notebooks
- `notebooks/exploratory/01_initial_exploration.ipynb` - Initial EDA

### API
- `src/api/app.py` - FastAPI application for model serving

### Testing
- `tests/test_data_processing.py` - Data processing tests

### DevOps
- `.github/workflows/ci-cd.yml` - CI/CD pipeline
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-container setup

---

## Common Commands Reference

```bash
# Development
make install-dev          # Install with dev dependencies
make format              # Format code with black
make lint                # Run linters
make test                # Run tests
make test-cov            # Run tests with coverage

# Data Pipeline
make data-download       # Download data
make data-process        # Process data

# Model Development
make train               # Train models
make notebook            # Start Jupyter

# Deployment
make serve               # Start API server
make docker-build        # Build Docker image
make docker-run          # Run with docker-compose

# Cleanup
make clean               # Remove cache and build files
```

---

## Team Workflow

### Daily Workflow
1. Pull latest changes: `git pull origin develop`
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test: `make test`
4. Format code: `make format`
5. Commit: `git commit -m "feat: your feature"`
6. Push and create PR: `git push origin feature/your-feature`

### Code Review Checklist
- [ ] Tests pass locally
- [ ] Code formatted (black, isort)
- [ ] Linting passes (flake8)
- [ ] Documentation updated
- [ ] PR description clear
- [ ] No large files committed

---

## Hypothesis Testing Plan

### Hypothesis
The growth of ride-hailing platforms significantly reduced traditional taxi trip volumes in Manhattan while increasing overall trip activity in outer boroughs, particularly during off-peak hours.

### Testing Approach

1. **Temporal Analysis**
   - Compare yellow/green taxi volumes before and after ride-hailing introduction
   - Use time series analysis to identify trend breaks
   - Statistical tests: Mann-Kendall trend test, Chow test

2. **Spatial Analysis**
   - Compare trip distributions across boroughs
   - Manhattan vs outer boroughs
   - Statistical tests: Chi-square test, Kolmogorov-Smirnov test

3. **Peak vs Off-Peak**
   - Define peak hours (7-9 AM, 5-7 PM)
   - Compare growth rates during different time periods
   - Statistical tests: T-test, ANOVA

---

## Next Steps

### Immediate (This Week)
1. Download data for 2020-2024
2. Complete initial data processing
3. Run exploratory analysis notebook
4. Document initial findings

### Short-term (This Month)
1. Complete EDA for all vehicle types
2. Develop baseline models
3. Begin hypothesis testing
4. Create initial visualizations

### Medium-term (Next 2-3 Months)
1. Develop advanced models
2. Complete hypothesis testing
3. Build API and dashboard
4. Write final report

---

## Resources

### Documentation
- [Project README](README.md)
- [Data Dictionary](docs/data_dictionary.md)
- [Contributing Guide](CONTRIBUTING.md)
- [API Documentation](http://localhost:8000/docs) (when running)

### External Resources
- [NYC TLC Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [NYC Open Data](https://opendata.cityofnewyork.us/)

### Learning Resources
- Time Series Analysis: statsmodels, prophet documentation
- Geospatial Analysis: geopandas, folium documentation
- MLOps: MLflow documentation

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
```bash
# Solution: Install package in development mode
pip install -e .
```

**Issue**: Out of memory when processing data
```bash
# Solution: Process data in chunks or use Dask
# Modify src/data/process_data.py to use chunking
```

**Issue**: Tests failing
```bash
# Solution: Check Python version and dependencies
python --version  # Should be 3.9+
pip install -r requirements.txt
```

---

## Contact and Support

- **Issues**: Open a GitHub issue
- **Questions**: Check documentation first, then ask team
- **Contributions**: See [CONTRIBUTING.md](CONTRIBUTING.md)
