.PHONY: help install install-dev test lint format clean data train forecast-borough forecast-borough-tree forecast-borough-tuned-stack forecast-slide-visuals serve docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install       : Install production dependencies"
	@echo "  install-dev   : Install development dependencies"
	@echo "  test          : Run tests"
	@echo "  test-cov      : Run tests with coverage"
	@echo "  lint          : Run linters (flake8, mypy)"
	@echo "  format        : Format code (black, isort)"
	@echo "  clean         : Remove build artifacts and cache"
	@echo "  data-download : Download NYC TLC data"
	@echo "  data-process  : Process raw data"
	@echo "  data-sample-yellow : Build representative yellow taxi sample"
	@echo "  forecast-borough : Train explainable borough forecast (next 7 days)"
	@echo "  forecast-borough-tree : Train XGBoost/LightGBM borough forecast models"
	@echo "  forecast-borough-tuned-stack : Tune tree models + stacking ensemble"
	@echo "  forecast-slide-visuals : Build forecast map + model comparison charts for slides"
	@echo "  train         : Train models"
	@echo "  serve         : Start API server"
	@echo "  docker-build  : Build Docker image"
	@echo "  docker-run    : Run Docker container"
	@echo "  notebook      : Start Jupyter notebook"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term tests/

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/

data-download:
	python src/data/download_data.py --start-year 2020 --end-year 2024

data-process:
	python src/data/process_data.py

data-sample-yellow:
	python src/data/sample_yellow_data.py --start-year 2009 --end-year 2025 --sample-size 24000 --output-dir data/samples

forecast-borough:
	python src/models/forecast_borough_demand.py --horizon-hours 168

forecast-borough-tree:
	python src/models/forecast_borough_tree_models.py --horizon-hours 168

forecast-borough-tuned-stack:
	python src/models/tune_and_stack_borough_models.py --horizon-hours 168 --tune-iter 6

forecast-slide-visuals:
	python src/models/build_forecast_slide_visuals.py

train:
	python src/models/train_model.py --config config/model_config.yaml

serve:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t nyc-mobility-analysis:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

notebook:
	jupyter notebook notebooks/

mlflow:
	mlflow ui --backend-store-uri file:./mlruns
