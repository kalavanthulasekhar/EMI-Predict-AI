# ==========================================
# EMI PREDICT-AI
# PROFESSIONAL STREAMLIT APPLICATION
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EMI Predict-AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_PATH = Path(__file__).resolve().parent.parent

CLASSIFICATION_PATH = (
    BASE_PATH / "models" / "final"
)

REGRESSION_PATH = (
    BASE_PATH / "models" / "regression"
)


# ==========================================
# LOAD CSS
# ==========================================

CSS_PATH = (
    Path(__file__).resolve().parent
    / "style.css"
)

if CSS_PATH.exists():

    with open(
        CSS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    
    classification_model = joblib.load(
        CLASSIFICATION_PATH /
        "emi_eligibility_classifier.pkl"
    )
    classification_preprocessor = joblib.load(
        CLASSIFICATION_PATH /
        "classification_preprocessor.pkl"
    )

    try:
        regression_model = joblib.load(
            REGRESSION_PATH /
            "gradient_boosting_regressor.pkl"
        )
    except (ModuleNotFoundError, ImportError):
        # Fallback to linear regression if gradient boosting fails
        regression_model = joblib.load(
            REGRESSION_PATH /
            "linear_regression.pkl"
        )
    
    regression_preprocessor = joblib.load(
        REGRESSION_PATH /
        "regression_preprocessor.pkl"
    )

    return (
        classification_model,
        classification_preprocessor,
        regression_model,
        regression_preprocessor
    )


(
    classification_model,
    classification_preprocessor,
    regression_model,
    regression_preprocessor
) = load_models()

# ==========================================
# APPLICATION NAVIGATION
# ==========================================

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Analysis Dashboard",
        "🔮 EMI Prediction"
    ]
)

# ==========================================
# LOAD DATASET FOR ANALYSIS
# ==========================================

@st.cache_data
def load_analysis_data():

    data_path = (
        BASE_PATH
        / "data"
        / "raw"
        / "emi_prediction_dataset.csv"
    )

    df = pd.read_csv(
        data_path,
        low_memory=False
    )

    return df

if page == "📊 Analysis Dashboard":

    df_analysis = load_analysis_data()

    # ==========================================
    # DASHBOARD HEADER
    # ==========================================

    st.markdown(
        '<div class="hero">'
        '<div class="badge">● DATA ANALYTICS CENTER</div>'
        '<div class="hero-title">📊 EMI Analysis Dashboard</div>'
        '<div class="hero-subtitle">Explore applicant demographics, financial patterns and EMI eligibility.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ==========================================
    # DATASET OVERVIEW
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '📌 Dataset Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Applicants",
            f"{len(df_analysis):,}",
            border=True
        )

    with col2:
        st.metric(
            "Features",
            "27",
            border=True
        )

    with col3:
        st.metric(
            "Eligible",
            "18.39%",
            border=True
        )

    with col4:
        st.metric(
            "Not Eligible",
            "77.29%",
            border=True
        )

    # ==========================================
    # ELIGIBILITY DISTRIBUTION
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '🎯 EMI Eligibility Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    eligibility_counts = (
        df_analysis[
            "emi_eligibility"
        ]
        .value_counts()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            eligibility_counts
        )

    with col2:

        eligibility_percent = (
            eligibility_counts
            / len(df_analysis)
            * 100
        )

        st.dataframe(
            pd.DataFrame({
                "Applicants": eligibility_counts,
                "Percentage": (
                    eligibility_percent
                    .round(2)
                    .astype(str)
                    + "%"
                )
            }),
            use_container_width=True
        )

    # ==========================================
    # APPLICANT DEMOGRAPHICS
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '👥 Applicant Demographics'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        gender_counts = (
            df_analysis["gender"]
            .value_counts()
        )

        st.write("### Gender Distribution")

        st.bar_chart(
            gender_counts
        )

    with col2:

        employment_counts = (
            df_analysis[
                "employment_type"
            ]
            .value_counts()
        )

        st.write("### Employment Type")

        st.bar_chart(
            employment_counts
        )

    col1, col2 = st.columns(2)

    with col1:

        education_counts = (
            df_analysis[
                "education"
            ]
            .value_counts()
        )

        st.write("### Education")

        st.bar_chart(
            education_counts
        )

    with col2:

        marital_counts = (
            df_analysis[
                "marital_status"
            ]
            .value_counts()
        )

        st.write("### Marital Status")

        st.bar_chart(
            marital_counts
        )

    # ==========================================
    # FINANCIAL ANALYSIS
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '💰 Financial Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    financial_df = df_analysis.copy()

    # Convert to numeric
    numeric_cols = [
        "monthly_rent",
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
        "current_emi_amount",
        "monthly_salary"
    ]
    
    for col in numeric_cols:
        if col in financial_df.columns:
            financial_df[col] = pd.to_numeric(
                financial_df[col],
                errors='coerce'
            )

    financial_df[
        "total_living_expenses"
    ] = (
        financial_df[
            [
                "monthly_rent",
                "school_fees",
                "college_fees",
                "travel_expenses",
                "groceries_utilities",
                "other_monthly_expenses"
            ]
        ]
        .sum(axis=1)
    )

    financial_df[
        "total_monthly_commitments"
    ] = (
        financial_df[
            "total_living_expenses"
        ]
        +
        financial_df[
            "current_emi_amount"
        ]
    )

    financial_df[
        "disposable_income"
    ] = (
        financial_df[
            "monthly_salary"
        ]
        -
        financial_df[
            "total_monthly_commitments"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Average Salary",
            f"₹{financial_df['monthly_salary'].mean():,.0f}",
            border=True
        )

    with col2:

        st.metric(
            "Average Expenses",
            f"₹{financial_df['total_living_expenses'].mean():,.0f}",
            border=True
        )

    with col3:

        st.metric(
            "Average Disposable Income",
            f"₹{financial_df['disposable_income'].mean():,.0f}",
            border=True
        )

    with col4:

        st.metric(
            "Average Current EMI",
            f"₹{financial_df['current_emi_amount'].mean():,.0f}",
            border=True
        )

    financial_summary = (
        financial_df
        .groupby("emi_eligibility")
        [
            [
                "monthly_salary",
                "total_living_expenses",
                "disposable_income",
                "current_emi_amount"
            ]
        ]
        .mean()
        .round(2)
    )

    st.write(
        "### Financial Profile by Eligibility"
    )

    st.dataframe(
        financial_summary,
        use_container_width=True
    )

    # ==========================================
    # EMI ANALYSIS
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '📈 EMI Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "### Maximum Monthly EMI Distribution"
        )

        st.line_chart(
            df_analysis[
                "max_monthly_emi"
            ]
            .sort_values()
            .reset_index(drop=True)
            .iloc[
                ::max(
                    1,
                    len(df_analysis) // 1000
                )
            ]
        )

    with col2:

        emi_by_eligibility = (
            df_analysis
            .groupby(
                "emi_eligibility"
            )[
                "max_monthly_emi"
            ]
            .mean()
            .round(2)
        )

        st.write(
            "### Average Maximum EMI by Eligibility"
        )

        st.bar_chart(
            emi_by_eligibility
        )

    # ==========================================
    # REQUESTED LOAN ANALYSIS
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "### Requested Loan Amount"
        )

        requested_summary = (
            df_analysis
            .groupby(
                "emi_eligibility"
            )[
                "requested_amount"
            ]
            .mean()
            .round(0)
        )

        st.bar_chart(
            requested_summary
        )

    with col2:

        st.write(
            "### Requested Tenure"
        )

        tenure_summary = (
            df_analysis
            .groupby(
                "emi_eligibility"
            )[
                "requested_tenure"
            ]
            .mean()
            .round(1)
        )

        st.bar_chart(
            tenure_summary
        )

    # ==========================================
    # KEY INSIGHTS
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '💡 Key Insights'
        '</div>',
        unsafe_allow_html=True
    )

    not_eligible_pct = (
        (
            df_analysis[
                "emi_eligibility"
            ]
            == "Not_Eligible"
        )
        .mean()
        * 100
    )

    eligible_pct = (
        (
            df_analysis[
                "emi_eligibility"
            ]
            == "Eligible"
        )
        .mean()
        * 100
    )

    high_risk_pct = (
        (
            df_analysis[
                "emi_eligibility"
            ]
            == "High_Risk"
        )
        .mean()
        * 100
    )

    avg_salary = (
        financial_df[
            "monthly_salary"
        ].mean()
    )

    avg_disposable = (
        financial_df[
            "disposable_income"
        ].mean()
    )

    st.info(
        f"""
        **Dataset Insights**

        • **{not_eligible_pct:.2f}%** of applicants
        are classified as Not Eligible.

        • **{eligible_pct:.2f}%** of applicants
        are classified as Eligible.

        • **{high_risk_pct:.2f}%** fall into the
        High Risk category.

        • Average monthly salary is approximately
        **₹{avg_salary:,.0f}**.

        • Average disposable income is approximately
        **₹{avg_disposable:,.0f}**.
        """
    )

elif page == "🔮 EMI Prediction":

    # ==========================================
    # FUNCTIONS
    # ==========================================

    def calculate_requested_emi(
        principal,
        annual_interest_rate,
        tenure_months
    ):

        monthly_rate = (
            annual_interest_rate / 12 / 100
        )

        if monthly_rate == 0:

            return principal / tenure_months

        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** tenure_months
        ) / (
            (1 + monthly_rate) ** tenure_months - 1
        )

        return emi


    def create_features(data):

        expense_columns = [
            "monthly_rent",
            "school_fees",
            "college_fees",
            "travel_expenses",
            "groceries_utilities",
            "other_monthly_expenses"
        ]

        data["total_living_expenses"] = (
            data[expense_columns].sum(axis=1)
        )

        data["total_monthly_commitments"] = (
            data["total_living_expenses"]
            + data["current_emi_amount"]
        )

        data["disposable_income"] = (
            data["monthly_salary"]
            - data["total_monthly_commitments"]
        )

        salary_safe = (
            data["monthly_salary"].replace(
                0,
                np.nan
            )
        )

        data["expense_to_income_ratio"] = (
            data["total_living_expenses"]
            / salary_safe
        )

        data["commitment_to_income_ratio"] = (
            data["total_monthly_commitments"]
            / salary_safe
        )

        data["current_emi_to_income_ratio"] = (
            data["current_emi_amount"]
            / salary_safe
        )

        data["requested_amount_to_income"] = (
            data["requested_amount"]
            / salary_safe
        )

        data["emergency_fund_to_income"] = (
            data["emergency_fund"]
            / salary_safe
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.fillna(0)

        return data


    # ==========================================
    # SIDEBAR
    # ==========================================

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-title">'
            '💰 EMI Predict-AI'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="small-text">'
            'AI-powered EMI eligibility assessment'
            '</div>',
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown("### 🤖 AI Models")

        st.caption(
            "Classification: Logistic Regression"
        )

        st.caption(
            "Regression: Gradient Boosting"
        )

        st.divider()

        st.markdown("### 📊 Test Performance")

        st.metric(
            "Classification Accuracy",
            "79.57%"
        )

        st.metric(
            "High-Risk Recall",
            "68.34%"
        )

        st.metric(
            "Regression R²",
            "98.84%"
        )

        st.divider()

        st.caption(
            "For educational and decision-support purposes."
        )


    # ==========================================
    # HERO HEADER
    # ==========================================

    hero_html = """
    <div class="hero">
        <div class="badge">● AI RISK ASSESSMENT ENGINE</div>
        <div class="hero-title">💰 EMI Predict-AI</div>
        <div class="hero-subtitle">Smart EMI Eligibility & Affordability Platform</div>
        <p>Analyze an applicant's financial profile, estimate maximum affordable EMI, and assess loan eligibility using machine learning.</p>
    </div>
    """

    st.markdown(hero_html, unsafe_allow_html=True)


    # ==========================================
    # APPLICANT PROFILE
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '👤 Applicant Profile'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Enter the applicant personal and employment information.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        age = st.number_input(
            "Age",
            min_value=26,
            max_value=59,
            value=38
        )


    with col2:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )


    with col3:

        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single"]
        )


    with col4:

        education = st.selectbox(
            "Education",
            [
                "Graduate",
                "Post Graduate",
                "High School",
                "Professional"
            ]
        )


    # ==========================================
    # EMPLOYMENT
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '💼 Employment Profile'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        monthly_salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=3967.0,
            max_value=499970.0,
            value=50000.0,
            step=1000.0
        )


    with col2:

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Private",
                "Government",
                "Self-employed"
            ]
        )


    with col3:

        years_of_employment = st.number_input(
            "Years of Employment",
            min_value=0.5,
            max_value=36.0,
            value=5.0,
            step=0.1
        )


    with col4:

        company_type = st.selectbox(
            "Company Type",
            [
                "Large Indian",
                "MNC",
                "Mid-size",
                "Startup",
                "Small"
            ]
        )


    # ==========================================
    # HOUSEHOLD
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '🏠 Household Information'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        house_type = st.selectbox(
            "House Type",
            [
                "Rented",
                "Own",
                "Family"
            ]
        )


    with col2:

        monthly_rent = st.number_input(
            "Monthly Rent (₹)",
            min_value=0.0,
            value=5000.0
        )


    with col3:

        family_size = st.number_input(
            "Family Size",
            min_value=1,
            max_value=5,
            value=3
        )


    with col4:

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=4,
            value=1
        )


    # ==========================================
    # FINANCIAL PROFILE
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '💰 Financial Profile'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        school_fees = st.number_input(
            "School Fees (₹)",
            min_value=0.0,
            value=3000.0
        )


    with col2:

        college_fees = st.number_input(
            "College Fees (₹)",
            min_value=0.0,
            value=0.0
        )


    with col3:

        travel_expenses = st.number_input(
            "Travel Expenses (₹)",
            min_value=0.0,
            value=5000.0
        )


    with col4:

        groceries_utilities = st.number_input(
            "Groceries & Utilities (₹)",
            min_value=0.0,
            value=10000.0
        )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0.0,
            value=5000.0
        )


    with col2:

        current_emi_amount = st.number_input(
            "Current EMI (₹)",
            min_value=0.0,
            value=0.0
        )


    with col3:

        credit_score = st.number_input(
            "Credit Score",
            min_value=407.0,
            max_value=850.0,
            value=700.0
        )


    with col4:

        existing_loans = st.selectbox(
            "Existing Loans",
            ["No", "Yes"]
        )


    col1, col2 = st.columns(2)


    with col1:

        bank_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=6100.0,
            max_value=1717300.0,
            value=100000.0
        )


    with col2:

        emergency_fund = st.number_input(
            "Emergency Fund (₹)",
            min_value=1400.0,
            max_value=891500.0,
            value=50000.0
        )


    # ==========================================
    # LOAN REQUEST
    # ==========================================

    st.markdown(
        '<div class="section-title">'
        '📋 Loan Request'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        emi_scenario = st.selectbox(
            "EMI Scenario",
            [
                "Home Appliances EMI",
                "Personal Loan EMI",
                "E-commerce Shopping EMI",
                "Education EMI",
                "Vehicle EMI"
            ]
        )


    with col2:

        requested_amount = st.number_input(
            "Requested Amount (₹)",
            min_value=10000.0,
            max_value=1500000.0,
            value=250000.0,
            step=10000.0
        )


    with col3:

        requested_tenure = st.number_input(
            "Requested Tenure (Months)",
            min_value=3,
            max_value=84,
            value=24
        )


    # ==========================================
    # ANALYZE APPLICATION
    # ==========================================

    st.markdown(
        """
        <div style="
            text-align:center;
            padding: 1.5rem 0 0.5rem 0;
        ">
            <h2>🔮 Ready to Analyze?</h2>
            <p style="opacity:0.65;">
                Review the applicant information and run the AI assessment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "🚀 Analyze Loan Application",
        type="primary",
        use_container_width=True
    )


    # ==========================================
    # EMPTY STATE MESSAGE
    # ==========================================

    if not predict_button:

        st.info(
            "👆 Enter the applicant details above "
            "and click **Analyze Loan Application** "
            "to generate the AI assessment."
        )


    # ==========================================
    # PREDICTION
    # ==========================================

    if predict_button:

        # --------------------------------------
        # INPUT VALIDATION
        # --------------------------------------

        if monthly_salary <= 0:

            st.error(
                "Monthly salary must be greater than zero."
            )

            st.stop()

        if requested_amount <= 0:

            st.error(
                "Requested amount must be greater than zero."
            )

            st.stop()

        if requested_tenure <= 0:

            st.error(
                "Requested tenure must be greater than zero."
            )

            st.stop()


        # --------------------------------------
        # CREATE INPUT DATA
        # --------------------------------------

        input_data = pd.DataFrame({

            "age": [age],

            "gender": [gender],

            "marital_status": [marital_status],

            "education": [education],

            "monthly_salary": [monthly_salary],

            "employment_type": [employment_type],

            "years_of_employment": [
                years_of_employment
            ],

            "company_type": [company_type],

            "house_type": [house_type],

            "monthly_rent": [monthly_rent],

            "family_size": [family_size],

            "dependents": [dependents],

            "school_fees": [school_fees],

            "college_fees": [college_fees],

            "travel_expenses": [
                travel_expenses
            ],

            "groceries_utilities": [
                groceries_utilities
            ],

            "other_monthly_expenses": [
                other_monthly_expenses
            ],

            "existing_loans": [
                existing_loans
            ],

            "current_emi_amount": [
                current_emi_amount
            ],

            "credit_score": [
                credit_score
            ],

            "bank_balance": [
                bank_balance
            ],

            "emergency_fund": [
                emergency_fund
            ],

            "emi_scenario": [
                emi_scenario
            ],

            "requested_amount": [
                requested_amount
            ],

            "requested_tenure": [
                requested_tenure
            ]

        })


        # --------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------

        input_data = create_features(
            input_data
        )


        # --------------------------------------
        # FINANCIAL CALCULATIONS
        # --------------------------------------

        total_living_expenses = float(
            input_data[
                "total_living_expenses"
            ].iloc[0]
        )

        total_monthly_commitments = float(
            input_data[
                "total_monthly_commitments"
            ].iloc[0]
        )

        disposable_income = float(
            input_data[
                "disposable_income"
            ].iloc[0]
        )

        expense_ratio = float(
            input_data[
                "expense_to_income_ratio"
            ].iloc[0]
        )

        commitment_ratio = float(
            input_data[
                "commitment_to_income_ratio"
            ].iloc[0]
        )


        # --------------------------------------
        # CLASSIFICATION
        # --------------------------------------

        classification_processed = (
            classification_preprocessor.transform(
                input_data
            )
        )

        eligibility_prediction = (
            classification_model.predict(
                classification_processed
            )[0]
        )


        # --------------------------------------
        # REGRESSION
        # --------------------------------------

        regression_processed = (
            regression_preprocessor.transform(
                input_data
            )
        )

        max_emi_prediction = (
            regression_model.predict(
                regression_processed
            )[0]
        )

        max_emi_prediction = max(
            0,
            max_emi_prediction
        )


        # --------------------------------------
        # REQUESTED EMI
        # --------------------------------------

        requested_emi = calculate_requested_emi(
            requested_amount,
            12.0,
            requested_tenure
        )


        # --------------------------------------
        # AFFORDABILITY
        # --------------------------------------

        if requested_emi <= max_emi_prediction:

            affordability_status = "Affordable"

        else:

            affordability_status = (
                "Above Recommended Limit"
            )


        # ======================================
        # RESULTS
        # ======================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🎯 AI Assessment'
            '</div>',
            unsafe_allow_html=True
        )


        # --------------------------------------
        # MAIN RESULT CARDS
        # --------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            if eligibility_prediction == "Eligible":

                st.success(
                    "🟢 ELIGIBLE"
                )

            elif eligibility_prediction == "High_Risk":

                st.warning(
                    "🟠 HIGH RISK"
                )

            else:

                st.error(
                    "🔴 NOT ELIGIBLE"
                )


        with col2:

            st.metric(
                "Maximum Monthly EMI",
                f"₹{max_emi_prediction:,.0f}",
                border=True
            )


        with col3:

            st.metric(
                "Estimated Requested EMI",
                f"₹{requested_emi:,.0f}",
                border=True
            )


        # --------------------------------------
        # FINANCIAL SUMMARY
        # --------------------------------------

        st.markdown(
            '<div class="section-title">'
            '💰 Financial Summary'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Monthly Income",
                f"₹{monthly_salary:,.0f}",
                border=True
            )


        with col2:

            st.metric(
                "Total Expenses",
                f"₹{total_living_expenses:,.0f}",
                border=True
            )


        with col3:

            st.metric(
                "Disposable Income",
                f"₹{disposable_income:,.0f}",
                border=True
            )


        with col4:

            st.metric(
                "Current EMI",
                f"₹{current_emi_amount:,.0f}",
                border=True
            )


        # --------------------------------------
        # FINANCIAL RATIOS
        # --------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📊 Financial Health'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Expense / Income",
                f"{expense_ratio * 100:.1f}%",
                border=True
            )


        with col2:

            st.metric(
                "Commitment / Income",
                f"{commitment_ratio * 100:.1f}%",
                border=True
            )


        with col3:

            emi_ratio = (
                requested_emi
                / max_emi_prediction * 100
                if max_emi_prediction > 0
                else 0
            )

            st.metric(
                "Requested / Maximum EMI",
                f"{emi_ratio:.1f}%",
                border=True
            )


        # --------------------------------------
        # AFFORDABILITY
        # --------------------------------------

        st.markdown(
            '<div class="section-title">'
            '⚖️ EMI Affordability'
            '</div>',
            unsafe_allow_html=True
        )


        if max_emi_prediction > 0:

            affordability_percentage = min(
                (
                    requested_emi
                    / max_emi_prediction
                ) * 100,
                100
            )

        else:

            affordability_percentage = 100


        if affordability_status == "Affordable":

            st.success(
                f"✅ Requested EMI of "
                f"₹{requested_emi:,.2f} is within "
                f"the predicted maximum EMI of "
                f"₹{max_emi_prediction:,.2f}."
            )

        else:

            difference = (
                requested_emi
                - max_emi_prediction
            )

            st.warning(
                f"⚠️ Requested EMI exceeds the "
                f"predicted maximum by "
                f"₹{difference:,.2f}."
            )


        st.progress(
            int(affordability_percentage)
        )


        # --------------------------------------
        # RISK EXPLANATION
        # --------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🚦 Risk Assessment'
            '</div>',
            unsafe_allow_html=True
        )


        if eligibility_prediction == "Eligible":

            st.success(
                "The applicant is classified as "
                "**Eligible** based on the supplied "
                "financial profile."
            )

        elif eligibility_prediction == "High_Risk":

            st.warning(
                "The applicant is classified as "
                "**High Risk** and may require "
                "additional financial review."
            )

        else:

            st.error(
                "The applicant is classified as "
                "**Not Eligible** based on the "
                "supplied information."
            )


        # --------------------------------------
        # RECOMMENDATION
        # --------------------------------------

        st.markdown(
            '<div class="section-title">'
            '💡 AI Recommendation'
            '</div>',
            unsafe_allow_html=True
        )


        if affordability_status == "Affordable":

            st.info(
                "The requested EMI is within the "
                "predicted maximum EMI range."
            )

        else:

            st.info(
                "Consider reducing the requested "
                "loan amount or increasing the "
                "requested tenure."
            )


    # ==========================================
    # FOOTER
    # ==========================================

    st.markdown(
        """
        <div class="footer">
            EMI Predict-AI • Machine Learning Based
            EMI Eligibility & Affordability System
            <br>
            Decision-support tool — not a substitute
            for professional financial assessment.
        </div>
        """,
        unsafe_allow_html=True
    )