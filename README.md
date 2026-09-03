<div align="center">

# EMI Predict-AI

### Intelligent EMI affordability and eligibility analysis

An interactive machine learning dashboard that turns applicant financial data into clear, practical loan insights.

<p>
	<a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" alt="Built with Streamlit"></a>
	<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
	<img src="https://img.shields.io/badge/ML-scikit--learn-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Machine learning with scikit-learn">
</p>

<p>
	<a href="#-features">Features</a> &nbsp;•&nbsp;
	<a href="#-quick-start">Quick start</a> &nbsp;•&nbsp;
	<a href="#-deploy-on-render">Deploy</a> &nbsp;•&nbsp;
	<a href="#-responsible-use">Responsible use</a>
</p>

</div>

<br>

> **A decision-support demo, not a lending decision engine.** Explore financial patterns, test hypothetical applicant profiles, and compare requested repayments with model-estimated affordability.

## ✦ Features

| Dashboard | Prediction |
| --- | --- |
| Eligibility distribution and applicant demographics | Profile-based eligibility classification |
| Financial patterns and disposable-income analysis | Maximum recommended monthly EMI |
| Requested loan amount and tenure trends | Requested EMI and affordability comparison |
| Dataset insights with interactive charts | Clear `Eligible`, `High_Risk`, or `Not Eligible` result |

## ◈ How It Works

```text
Applicant profile  ->  Feature engineering  ->  Classification
											 \->  EMI regression  ->  Affordability comparison
```

1. Enter applicant, employment, household, financial, and loan-request details.
2. Generate derived features such as disposable income and income ratios.
3. Classify the application as `Eligible`, `High_Risk`, or `Not Eligible`.
4. Estimate the maximum monthly EMI with the trained regression model.
5. Calculate the requested EMI using the app's 12% annual-interest assumption.
6. Compare the requested EMI with the estimated recommended limit.

## ◎ Model Snapshot

| Validation metric | Result |
| --- | ---: |
| Classification accuracy | **79.57%** |
| High-risk recall | **68.34%** |
| Regression R2 | **98.84%** |

The app loads serialized models and preprocessors from `models/` at startup. These figures describe this project's validation results and do not guarantee real-world lending performance.

## ⚙ Technology

`Python` · `Streamlit` · `pandas` · `NumPy` · `scikit-learn` · `joblib`

## ◫ Project Structure

```text
EMIPredict-AI/
├── app/
│   ├── app.py                 # Streamlit application
│   └── style.css              # Custom interface styles
├── data/
│   ├── raw/                   # Source dataset
│   └── processed/             # Cleaned, engineered, and validation data
├── models/
│   ├── final/                 # Eligibility model and preprocessor
│   └── regression/            # EMI regressors and preprocessor
├── notebooks/                 # EDA, feature engineering, training, and validation
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/EMIPredict-AI.git
cd EMIPredict-AI
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install and launch

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

## ☁ Deploy On Render

Create a **Web Service** connected to this GitHub repository and use:

| Setting | Value |
| --- | --- |
| Environment | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app/app.py --server.address 0.0.0.0 --server.port $PORT` |

No secret environment variables are required by the current application. Keep `models/`, `data/`, and `app/` in the GitHub repository. The local `.venv/` directory is excluded by `.gitignore`.

## ⚠ Responsible Use

This project is intended for learning, experimentation, and financial analysis demonstrations. It is not financial advice, a credit bureau, or a substitute for responsible underwriting, regulatory review, or human judgment. Do not use the output as the sole basis for approving or rejecting a real loan application.

## License & Status

**License:** No license has been specified for this repository yet.

**Status:** Ready for local use and Render deployment.