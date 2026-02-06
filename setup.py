from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nyc_mobility_analysis",
    version="0.1.0",
    author="Your Team Name",
    author_email="your-email@example.com",
    description="Analysis of ride-hailing impact on NYC urban mobility patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nyc-mobility-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/nyc-mobility-analysis/issues",
        "Documentation": "https://github.com/yourusername/nyc-mobility-analysis/blob/main/README.md",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "geopandas>=0.13.0",
        "statsmodels>=0.14.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "ml": [
            "mlflow>=2.5.0",
            "wandb>=0.15.0",
            "optuna>=3.2.0",
            "prophet>=1.1.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "download-data=src.data.download_data:main",
            "process-data=src.data.process_data:main",
        ],
    },
)
