# NBC Nexus: AI-Powered Autonomous Aftermarket Platform

## 1. Project Overview
**NBC Nexus** is a prototype for a "Smart Aftermarket" ecosystem. It transforms the traditional bearing supply chain into a connected, data-driven loop. By integrating IoT telemetry, AI marketing, and circular economy principles, it ensures zero downtime for customers and sustainable revenue for NBC Bearings.

## 2. Architecture & Modules
The system is composed of an interactive Dashboard and a High-Performance API.

### Dashboard (`dashboard.py`)
A Streamlit-based "Control Tower" divided into three autonomous modules:

*   **IoT & Predictive Maintenance ("Digital Twin"):**
    *   Real-time health monitoring of industrial assets.
    *   Interactive "What-If" simulator (RPM, Temp, Vibration) to test AI predictions.
    *   Automatic triggering of replacement orders when Remaining Useful Life (RUL) < 30 days.

*   **Dealer Opportunity Engine ("Intelligent Router"):**
    *   Routes orders to the best dealers based on performance (Gold/Silver/Bronze clusters).
    *   Uses AI clustering to match urgency with dealer responsiveness and stock.

*   **Circular Economy Hub ("Sustainability Loop"):**
    *   Tracks CO2 abatement and asset recovery.
    *   AI-assisted inspection workflow for used bearings.

### NBC Analytics API (`api/`)
A dedicated FastAPI service hosting the machine learning models.
*   **Endpoints:**
    *   `/predict/rul`: RUL Prediction based on Random Forest.
    *   `/predict/dealer-segment`: Dealer segmentation using K-Means Clustering.
*   **Deployment:** Dockerized for easy cloud deployment (Hugging Face Spaces).

## 3. Data & AI Models

### Machine Learning Models
*   **RUL Predictor:** Random Forest Regressor trained on physics-based synthetic telemetry. Validates wear based on vibration, temp, and load.
*   **Dealer Segmenter:** K-Means Clustering to categorize partners into 'Premium', 'Standard', and 'At-Risk' based on CSI, turnaround time, and inventory.

### Synthetic Data (`data_gen.py`)
*   **Bearings Data:** 200 assets with physics-based failure modes (Lubrication failure, race defects).
*   **Dealer Data:** 20 network partners with performance metrics.

## 4. Installation & Setup

### Prerequisites
*   Python 3.9+
*   FastAPI & Uvicorn (for API)
*   Streamlit (for Dashboard)

### Running the API
The API serves the prediction models.

```bash
cd api
# Install dependencies
pip install -r requirements.txt
# Run the Server
uvicorn main:app --reload
```
API docs available at: `http://localhost:8000/docs`

### Running the Dashboard
The dashboard connects to the API (default: https://vivek45537-nbc.hf.space). To use a local API, update `API_URL` in `dashboard.py`.

```bash
# From the root folder
pip install -r requirements.txt
streamlit run dashboard.py
```

## 5. Docker Support
The API folder includes a `Dockerfile` for containerized deployment.

```bash
cd api
docker build -t nbc-nexus-api .
docker run -p 7860:7860 nbc-nexus-api
```
