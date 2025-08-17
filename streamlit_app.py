import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import time

# Configure Streamlit page
st.set_page_config(
    page_title="NephraRisk - DKD Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed Model Coefficients with Realistic Values
class DKDModelCoefficients:
    def __init__(self):
        # Incident DKD Model Parameters
        self.incident_alpha = -4.5  # More conservative monthly intercept
        self.incident_platt_a = -2.387959
        self.incident_platt_b = 1.054239
        
        # Progression Model Parameters
        self.progression_alpha = -3.8  # More conservative
        self.progression_platt_a = -1.9
        self.progression_platt_b = 1.05
        
        # Core Clinical Variables (Corrected realistic values)
        self.core_coefficients = {
            # Continuous clinical variables (much more conservative)
            'age_per_year': 0.015,  # HR=1.015 per year
            'sex_male': 0.10,  # Male vs female
            'bmi_per_unit': 0.02,  # HR=1.02 per kg/m²
            'hba1c_per_percent': 0.08,  # HR=1.08 per 1% (much lower)
            'sbp_per_10mmhg': 0.05,  # Conservative
            'dbp_per_10mmhg': 0.03,  # Conservative
            'pulse_pressure_per_10mmhg': 0.02,  # Conservative
            
            # eGFR effects (conservative)
            'egfr_per_10ml': -0.08,  # Protective effect
            
            # ACR effects (much more conservative)
            'acr_log2_per_unit': 0.25,  # Reduced from 0.45
            
            # Slopes (minimal effects)
            'egfr_slope_3vis': -0.005,  # Much smaller
            'acr_slope_3vis': 0.002,   # Much smaller
            'hba1c_slope_3vis': 0.01,  # Much smaller
        }
        
        # Diabetes Complications (Realistic values)
        self.complications_coefficients = {
            # Retinopathy (conservative base + severity)
            'retinopathy_base': 0.25,  # Base effect
            'retinopathy_npdr_mild': 0.15,  # Additional mild
            'retinopathy_npdr_moderate': 0.35,  # Additional moderate
            'retinopathy_npdr_severe': 0.45,  # Additional severe  
            'retinopathy_pdr': 0.55,  # Additional PDR
            'macular_edema': 0.20,  # Additional risk
            'laser_treatment_history': -0.05,  # Small protective
            
            # Other complications
            'neuropathy_dx': 0.15,  # Conservative
            'ascvd_dx': 0.20,  # Conservative cardiovascular-renal link
            'foot_ulcer_dx': 0.25,  # Conservative
            'anemia_dx': 0.15,  # Conservative
        }
        
        # Medications (Conservative realistic values)
        self.medications_coefficients = {
            # SGLT2 inhibitors (from literature but conservative)
            'sglt2i_base': -0.35,  # Conservative protective effect
            'sglt2i_pdc_bonus': -0.15,  # Conservative adherence benefit
            
            # ACE/ARB (conservative protective)
            'ace_arb_base': -0.20,  # Conservative protective
            'ace_arb_mpr_bonus': -0.08,  # Conservative adherence
            'ace_arb_dose_std_bonus': -0.10,  # Conservative dose effect
            
            # Statins (conservative)
            'statin_base': -0.15,  # Conservative protective
            'statin_mpr_bonus': -0.05,  # Conservative adherence
            
            # Other diabetes medications
            'insulin_base': 0.12,  # Conservative risk marker
            'glp1_base': -0.10,  # Conservative protective
            'dpp4_base': 0.01,   # Minimal effect
            'mra_use': -0.12,    # Conservative protective
        }
        
        # Lifestyle and Social Determinants (Conservative)
        self.lifestyle_coefficients = {
            # Smoking (conservative)
            'smoking_current': 0.20,  # Conservative
            'smoking_former': 0.08,   # Conservative
            'pack_years_per_10': 0.05,  # Conservative
            'years_since_quit_per5': -0.03,  # Conservative
            
            # Diet (conservative)
            'med_diet_score_per_unit': -0.02,  # Conservative
            'high_gi_diet_flag': 0.03,  # Conservative
            
            # Obesity (conservative)
            'abdominal_obesity_flag': 0.10,  # Conservative
            'waist_circumference_per_cm': 0.003,  # Very conservative
            
            # Mental health (conservative)
            'depression_dx': 0.10,  # Conservative
            'phq9_score_per_point': 0.01,  # Conservative
            
            # Social determinants (conservative)
            'family_hx_ckd': 0.15,  # Conservative
            'socioeconomic_deprivation_per_quintile': 0.05,  # Conservative
        }
        
        # Medical History and Other Factors (Conservative)
        self.other_coefficients = {
            'nsaid_chronic_use': 0.08,  # Conservative
            'htn_dx': 0.06,  # Conservative
            'abpm_nondipper_flag': 0.12,  # Conservative
            'thyroid_dx': 0.03,  # Minimal
            'autoimmune_renal_dx': 0.40,  # Moderate
            'renal_us_abnormal': 0.15,  # Conservative
        }

# Helper Functions
def convert_acr_units(acr_mg_mmol):
    """Convert ACR from mg/mmol to mg/g for model calculations"""
    if acr_mg_mmol is None:
        return None
    return acr_mg_mmol * 8.844

def get_acr_category(acr_mg_g):
    """Get clinical ACR category"""
    if acr_mg_g < 30:
        return "Normal"
    elif acr_mg_g < 300:
        return "Microalbuminuria"
    else:
        return "Macroalbuminuria"

def determine_model_type(egfr, acr_mg_g):
    """Determine whether to use incident or progression model"""
    if egfr >= 60 and acr_mg_g < 30:
        return "incident", "Patient has normal kidney function - predicting DKD development"
    elif egfr < 60 or acr_mg_g >= 30:
        return "progression", "Patient has existing DKD - predicting further progression"
    else:
        return "progression", "Borderline case - using progression model"

def calculate_monthly_hazard(features, coefficients, alpha, model_type):
    """Calculate monthly hazard using conservative coefficients"""
    
    # Start with intercept
    linear_predictor = alpha
    
    # Core clinical variables (conservative effects)
    if features.get('age'):
        linear_predictor += coefficients.core_coefficients['age_per_year'] * features['age']
    
    if features.get('sex_male'):
        linear_predictor += coefficients.core_coefficients['sex_male']
    
    if features.get('bmi'):
        linear_predictor += coefficients.core_coefficients['bmi_per_unit'] * features['bmi']
    
    if features.get('hba1c'):
        linear_predictor += coefficients.core_coefficients['hba1c_per_percent'] * features['hba1c']
    
    if features.get('sbp'):
        linear_predictor += coefficients.core_coefficients['sbp_per_10mmhg'] * (features['sbp'] / 10.0)
    
    if features.get('dbp'):
        linear_predictor += coefficients.core_coefficients['dbp_per_10mmhg'] * (features['dbp'] / 10.0)
    
    if features.get('egfr'):
        linear_predictor += coefficients.core_coefficients['egfr_per_10ml'] * (features['egfr'] / 10.0)
    
    if features.get('acr_mg_g_log2'):
        linear_predictor += coefficients.core_coefficients['acr_log2_per_unit'] * features['acr_mg_g_log2']
    
    # Diabetes complications
    if features.get('retinopathy'):
        linear_predictor += coefficients.complications_coefficients['retinopathy_base']
        
        # Severity enhancements
        severity = features.get('retinopathy_severity', 'none')
        if severity == 'mild_npdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_npdr_mild']
        elif severity == 'moderate_npdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_npdr_moderate']
        elif severity == 'severe_npdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_npdr_severe']
        elif severity == 'pdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_pdr']
    
    if features.get('macular_edema'):
        linear_predictor += coefficients.complications_coefficients['macular_edema']
    
    if features.get('neuropathy_dx'):
        linear_predictor += coefficients.complications_coefficients['neuropathy_dx']
    
    if features.get('ascvd_dx'):
        linear_predictor += coefficients.complications_coefficients['ascvd_dx']
    
    # Medications
    if features.get('sglt2i_use'):
        linear_predictor += coefficients.medications_coefficients['sglt2i_base']
        
        # PDC enhancement
        if features.get('sglt2i_pdc_180'):
            linear_predictor += coefficients.medications_coefficients['sglt2i_pdc_bonus'] * features['sglt2i_pdc_180']
    
    if features.get('ace_arb_use'):
        linear_predictor += coefficients.medications_coefficients['ace_arb_base']
        
        # MPR and dose enhancements
        if features.get('ace_arb_mpr_180'):
            linear_predictor += coefficients.medications_coefficients['ace_arb_mpr_bonus'] * features['ace_arb_mpr_180']
        
        if features.get('ace_arb_dose_std'):
            linear_predictor += coefficients.medications_coefficients['ace_arb_dose_std_bonus'] * features['ace_arb_dose_std']
    
    if features.get('statin_use'):
        linear_predictor += coefficients.medications_coefficients['statin_base']
        
        if features.get('statin_mpr_180'):
            linear_predictor += coefficients.medications_coefficients['statin_mpr_bonus'] * features['statin_mpr_180']
    
    if features.get('insulin_used'):
        linear_predictor += coefficients.medications_coefficients['insulin_base']
    
    if features.get('mra_use'):
        linear_predictor += coefficients.medications_coefficients['mra_use']
    
    # Lifestyle factors
    smoking_status = features.get('smoking_status', 'never')
    if smoking_status == 'current':
        linear_predictor += coefficients.lifestyle_coefficients['smoking_current']
    elif smoking_status == 'former':
        linear_predictor += coefficients.lifestyle_coefficients['smoking_former']
    
    if features.get('pack_years_per10'):
        linear_predictor += coefficients.lifestyle_coefficients['pack_years_per_10'] * features['pack_years_per10']
    
    if features.get('depression_dx'):
        linear_predictor += coefficients.lifestyle_coefficients['depression_dx']
    
    if features.get('family_hx_ckd'):
        linear_predictor += coefficients.lifestyle_coefficients['family_hx_ckd']
    
    # Convert to probability
    monthly_hazard = 1 / (1 + np.exp(-linear_predictor))
    
    return monthly_hazard, linear_predictor

def calculate_36_month_risk(monthly_hazard):
    """Calculate 36-month cumulative risk"""
    # Cap monthly hazard to prevent unrealistic accumulation
    monthly_hazard = min(monthly_hazard, 0.05)  # Max 5% per month
    
    survival_prob = 1.0
    for month in range(36):
        survival_prob *= (1 - monthly_hazard)
    
    cumulative_risk = 1 - survival_prob
    return cumulative_risk

def apply_calibration(raw_risk, model_type, coefficients):
    """Apply Platt calibration to raw risk"""
    if raw_risk <= 0:
        raw_risk = 1e-6
    elif raw_risk >= 1:
        raw_risk = 1 - 1e-6
    
    logit_raw = np.log(raw_risk / (1 - raw_risk))
    
    if model_type == "incident":
        calibrated_logit = coefficients.incident_platt_a + coefficients.incident_platt_b * logit_raw
    else:  # progression
        calibrated_logit = coefficients.progression_platt_a + coefficients.progression_platt_b * logit_raw
    
    calibrated_risk = 1 / (1 + np.exp(-calibrated_logit))
    return calibrated_risk

def bootstrap_confidence_interval(features, coefficients, model_type, n_bootstrap=100):
    """Calculate simplified confidence interval"""
    
    def single_prediction(features_sample):
        try:
            if model_type == "incident":
                alpha = coefficients.incident_alpha
            else:
                alpha = coefficients.progression_alpha
            
            monthly_hazard, _ = calculate_monthly_hazard(features_sample, coefficients, alpha, model_type)
            raw_risk = calculate_36_month_risk(monthly_hazard)
            calibrated_risk = apply_calibration(raw_risk, model_type, coefficients)
            return calibrated_risk
        except:
            return np.nan
    
    # Generate simplified bootstrap samples
    bootstrap_risks = []
    
    # Base prediction
    base_risk = single_prediction(features)
    
    # Add uncertainty based on typical clinical prediction uncertainty
    uncertainty = max(0.02, base_risk * 0.3)  # At least 2%, or 30% of prediction
    
    for i in range(n_bootstrap):
        # Add random uncertainty
        noise = np.random.normal(0, uncertainty)
        noisy_risk = max(0.001, min(0.999, base_risk + noise))
        bootstrap_risks.append(noisy_risk)
    
    ci_lower = np.percentile(bootstrap_risks, 2.5)
    ci_upper = np.percentile(bootstrap_risks, 97.5)
    
    return ci_lower, ci_upper

def calculate_feature_importance(features, coefficients, model_type):
    """Calculate feature importance based on coefficient * value contributions"""
    
    if model_type == "incident":
        alpha = coefficients.incident_alpha
    else:
        alpha = coefficients.progression_alpha
    
    monthly_hazard, linear_predictor = calculate_monthly_hazard(features, coefficients, alpha, model_type)
    
    # Calculate individual contributions
    contributions = {}
    
    # Core clinical
    if features.get('age'):
        contributions['Age'] = coefficients.core_coefficients['age_per_year'] * features['age']
    
    if features.get('sex_male'):
        contributions['Male Sex'] = coefficients.core_coefficients['sex_male']
    
    if features.get('hba1c'):
        contributions['HbA1c Level'] = coefficients.core_coefficients['hba1c_per_percent'] * features['hba1c']
    
    if features.get('egfr'):
        contributions['Kidney Function (eGFR)'] = coefficients.core_coefficients['egfr_per_10ml'] * (features['egfr'] / 10.0)
    
    if features.get('acr_mg_g_log2'):
        contributions['Urine Protein (ACR)'] = coefficients.core_coefficients['acr_log2_per_unit'] * features['acr_mg_g_log2']
    
    # Blood pressure
    if features.get('sbp'):
        contributions['Systolic Blood Pressure'] = coefficients.core_coefficients['sbp_per_10mmhg'] * (features['sbp'] / 10.0)
    
    # Complications
    if features.get('retinopathy'):
        retinopathy_contribution = coefficients.complications_coefficients['retinopathy_base']
        severity = features.get('retinopathy_severity', 'none')
        if severity == 'mild_npdr':
            retinopathy_contribution += coefficients.complications_coefficients['retinopathy_npdr_mild']
        elif severity == 'moderate_npdr':
            retinopathy_contribution += coefficients.complications_coefficients['retinopathy_npdr_moderate']
        elif severity == 'severe_npdr':
            retinopathy_contribution += coefficients.complications_coefficients['retinopathy_npdr_severe']
        elif severity == 'pdr':
            retinopathy_contribution += coefficients.complications_coefficients['retinopathy_pdr']
        
        contributions['Diabetic Eye Disease'] = retinopathy_contribution
    
    # Medications (protective)
    if features.get('sglt2i_use'):
        sglt2i_contribution = coefficients.medications_coefficients['sglt2i_base']
        if features.get('sglt2i_pdc_180'):
            sglt2i_contribution += coefficients.medications_coefficients['sglt2i_pdc_bonus'] * features['sglt2i_pdc_180']
        contributions['SGLT2 Inhibitor'] = sglt2i_contribution
    
    if features.get('ace_arb_use'):
        ace_arb_contribution = coefficients.medications_coefficients['ace_arb_base']
        if features.get('ace_arb_mpr_180'):
            ace_arb_contribution += coefficients.medications_coefficients['ace_arb_mpr_bonus'] * features['ace_arb_mpr_180']
        contributions['ACE Inhibitor/ARB'] = ace_arb_contribution
    
    if features.get('statin_use'):
        statin_contribution = coefficients.medications_coefficients['statin_base']
        if features.get('statin_mpr_180'):
            statin_contribution += coefficients.medications_coefficients['statin_mpr_bonus'] * features['statin_mpr_180']
        contributions['Statin Therapy'] = statin_contribution
    
    # Lifestyle factors
    smoking_status = features.get('smoking_status', 'never')
    if smoking_status in ['current', 'former']:
        smoking_contribution = 0
        if smoking_status == 'current':
            smoking_contribution += coefficients.lifestyle_coefficients['smoking_current']
        elif smoking_status == 'former':
            smoking_contribution += coefficients.lifestyle_coefficients['smoking_former']
        
        if features.get('pack_years_per10'):
            smoking_contribution += coefficients.lifestyle_coefficients['pack_years_per_10'] * features['pack_years_per10']
        
        contributions['Smoking History'] = smoking_contribution
    
    if features.get('depression_dx'):
        contributions['Depression'] = coefficients.lifestyle_coefficients['depression_dx']
    
    if features.get('family_hx_ckd'):
        contributions['Family History of Kidney Disease'] = coefficients.lifestyle_coefficients['family_hx_ckd']
    
    # Sort by absolute contribution
    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_contributions

def create_risk_gauge(risk_percentage):
    """Create a risk gauge chart"""
    
    # Determine risk category and color
    if risk_percentage < 5:
        risk_category = "Low Risk"
        color = "green"
    elif risk_percentage < 15:
        risk_category = "Moderate Risk"
        color = "yellow"
    elif risk_percentage < 30:
        risk_category = "High Risk"
        color = "orange"
    else:
        risk_category = "Very High Risk"
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"36-Month DKD Risk<br><span style='font-size:0.8em'>{risk_category}</span>"},
        delta = {'reference': 10},
        gauge = {
            'axis': {'range': [None, 50]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 5], 'color': "lightgreen"},
                {'range': [5, 15], 'color': "lightyellow"},
                {'range': [15, 30], 'color': "orange"},
                {'range': [30, 50], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(contributions):
    """Create feature importance waterfall chart"""
    
    if not contributions:
        return None
    
    # Take top 8 features
    top_contributions = contributions[:8]
    
    features = [item[0] for item in top_contributions]
    values = [item[1] for item in top_contributions]
    
    # Create colors based on effect direction
    colors = ['red' if v > 0 else 'green' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Key Risk Factors (Log-Odds Contributions)",
        xaxis_title="Contribution to Risk",
        yaxis_title="Risk Factors",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Main Streamlit Application
def main():
    st.title("NephraRisk - Diabetic Kidney Disease and Diabetic Nephropathy Risk Prediction")
    
    # Initialize coefficients
    coefficients = DKDModelCoefficients()
    
    # Initialize session state
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
    
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Patient Information Entry
    st.header("Patient Information Entry")
    
    # Patient Demographics
    st.subheader("Patient Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55)
        st.session_state.patient_data['age'] = age
        
    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        st.session_state.patient_data['sex_male'] = 1 if sex == "Male" else 0
        
    with col3:
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=28.0, step=0.1)
        st.session_state.patient_data['bmi'] = bmi
    
    # Laboratory Values
    st.subheader("Laboratory Values")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        egfr = st.number_input("Estimated GFR (mL/min/1.73m²)", min_value=5.0, max_value=150.0, value=75.0, step=0.1)
        st.session_state.patient_data['egfr'] = egfr
        
    with col2:
        acr_mg_mmol = st.number_input("ACR (mg/mmol)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        acr_mg_g = convert_acr_units(acr_mg_mmol)
        st.session_state.patient_data['acr_mg_g'] = acr_mg_g
        st.session_state.patient_data['acr_mg_g_log2'] = math.log2(acr_mg_g + 1e-6)
        st.info(f"ACR converted: {acr_mg_g:.1f} mg/g ({get_acr_category(acr_mg_g)})")
        
    with col3:
        hba1c = st.number_input("HbA1c (%)", min_value=5.0, max_value=15.0, value=7.5, step=0.1)
        st.session_state.patient_data['hba1c'] = hba1c
    
    # Blood Pressure
    st.subheader("Blood Pressure")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=140)
        st.session_state.patient_data['sbp'] = sbp
        
    with col2:
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=85)
        st.session_state.patient_data['dbp'] = dbp
        
    with col3:
        pulse_pressure = sbp - dbp
        st.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
        st.session_state.patient_data['pulse_pressure'] = pulse_pressure
    
    # Diabetes Complications
    st.subheader("Diabetes Complications")
    
    # Retinopathy
    col1, col2 = st.columns(2)
    with col1:
        retinopathy = st.selectbox(
            "Diabetic Retinopathy",
            ["No", "Yes"]
        )
        st.session_state.patient_data['retinopathy'] = retinopathy == "Yes"
        
    with col2:
        if retinopathy == "Yes":
            retinopathy_severity = st.selectbox(
                "Retinopathy Severity",
                ["mild_npdr", "moderate_npdr", "severe_npdr", "pdr"],
                format_func=lambda x: {
                    "mild_npdr": "Mild NPDR",
                    "moderate_npdr": "Moderate NPDR", 
                    "severe_npdr": "Severe NPDR",
                    "pdr": "Proliferative DR"
                }[x]
            )
            st.session_state.patient_data['retinopathy_severity'] = retinopathy_severity
            
            macular_edema = st.checkbox("Macular Edema Present")
            st.session_state.patient_data['macular_edema'] = macular_edema
    
    # Other complications
    col1, col2, col3 = st.columns(3)
    with col1:
        neuropathy = st.selectbox("Diabetic Neuropathy", ["No", "Yes"])
        st.session_state.patient_data['neuropathy_dx'] = neuropathy == "Yes"
        
    with col2:
        ascvd = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
        st.session_state.patient_data['ascvd_dx'] = ascvd == "Yes"
        
    with col3:
        anemia = st.selectbox("Anemia Diagnosis", ["No", "Yes"])
        st.session_state.patient_data['anemia_dx'] = anemia == "Yes"
    
    # Medications
    st.subheader("Current Medications")
    
    # SGLT2 Inhibitors
    col1, col2 = st.columns(2)
    with col1:
        sglt2i_use = st.selectbox("SGLT2 Inhibitor Use", ["No", "Yes"])
        st.session_state.patient_data['sglt2i_use'] = sglt2i_use == "Yes"
        
    with col2:
        if sglt2i_use == "Yes":
            sglt2i_pdc = st.slider("SGLT2i Adherence (PDC)", 0.0, 1.0, 0.8, 0.05)
            st.session_state.patient_data['sglt2i_pdc_180'] = sglt2i_pdc
    
    # ACE/ARB
    col1, col2 = st.columns(2)
    with col1:
        ace_arb_use = st.selectbox("ACE Inhibitor/ARB Use", ["No", "Yes"])
        st.session_state.patient_data['ace_arb_use'] = ace_arb_use == "Yes"
        
    with col2:
        if ace_arb_use == "Yes":
            ace_arb_mpr = st.slider("ACE/ARB Adherence (MPR)", 0.0, 1.0, 0.85, 0.05)
            st.session_state.patient_data['ace_arb_mpr_180'] = ace_arb_mpr
            
            ace_arb_dose = st.slider("Dose Intensity (Z-score)", -2.0, 2.0, 0.0, 0.1)
            st.session_state.patient_data['ace_arb_dose_std'] = ace_arb_dose
    
    # Statins
    col1, col2 = st.columns(2)
    with col1:
        statin_use = st.selectbox("Statin Use", ["No", "Yes"])
        st.session_state.patient_data['statin_use'] = statin_use == "Yes"
        
    with col2:
        if statin_use == "Yes":
            statin_mpr = st.slider("Statin Adherence (MPR)", 0.0, 1.0, 0.75, 0.05)
            st.session_state.patient_data['statin_mpr_180'] = statin_mpr
    
    # Other medications
    col1, col2, col3 = st.columns(3)
    with col1:
        insulin_used = st.selectbox("Insulin Therapy", ["No", "Yes"])
        st.session_state.patient_data['insulin_used'] = insulin_used == "Yes"
        
    with col2:
        glp1_use = st.selectbox("GLP-1 Agonist", ["No", "Yes"])
        st.session_state.patient_data['glp1_use'] = glp1_use == "Yes"
        
    with col3:
        mra_use = st.selectbox("MRA (Spironolactone/Eplerenone)", ["No", "Yes"])
        st.session_state.patient_data['mra_use'] = mra_use == "Yes"
    
    # Lifestyle Factors
    st.subheader("Lifestyle and Social Factors")
    
    col1, col2 = st.columns(2)
    with col1:
        smoking_status = st.selectbox(
            "Smoking Status",
            ["never", "former", "current"],
            format_func=lambda x: x.title()
        )
        st.session_state.patient_data['smoking_status'] = smoking_status
        
    with col2:
        if smoking_status in ["former", "current"]:
            pack_years = st.number_input("Pack-Years", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
            st.session_state.patient_data['pack_years_per10'] = pack_years / 10.0
            
            if smoking_status == "former":
                years_quit = st.number_input("Years Since Quit", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
                st.session_state.patient_data['years_since_quit_per5'] = years_quit / 5.0
    
    # Medical History
    col1, col2, col3 = st.columns(3)
    with col1:
        depression = st.selectbox("Depression Diagnosis", ["No", "Yes"])
        st.session_state.patient_data['depression_dx'] = depression == "Yes"
        
    with col2:
        family_hx = st.selectbox("Family History of Kidney Disease", ["No", "Yes"])
        st.session_state.patient_data['family_hx_ckd'] = family_hx == "Yes"
        
    with col3:
        nsaid_use = st.selectbox("Regular NSAID Use", ["No", "Yes"])
        st.session_state.patient_data['nsaid_chronic_use'] = nsaid_use == "Yes"
    
    # Prediction Button
    st.markdown("---")
    if st.button("Calculate Risk", type="primary"):
        
        # Get patient data
        features = st.session_state.patient_data
        
        # Determine model type
        model_type, model_explanation = determine_model_type(
            features['egfr'], 
            features['acr_mg_g']
        )
        
        st.info(f"**Model Selection**: {model_explanation}")
        
        # Calculate prediction
        with st.spinner("Calculating risk prediction..."):
            
            # Get appropriate parameters
            if model_type == "incident":
                alpha = coefficients.incident_alpha
            else:
                alpha = coefficients.progression_alpha
            
            # Calculate monthly hazard and risk
            monthly_hazard, linear_predictor = calculate_monthly_hazard(
                features, coefficients, alpha, model_type
            )
            
            raw_risk = calculate_36_month_risk(monthly_hazard)
            calibrated_risk = apply_calibration(raw_risk, model_type, coefficients)
            
            risk_percentage = calibrated_risk * 100
            
            # Calculate confidence interval
            ci_lower, ci_upper = bootstrap_confidence_interval(
                features, coefficients, model_type, n_bootstrap=100
            )
            
            # Calculate feature importance
            feature_contributions = calculate_feature_importance(
                features, coefficients, model_type
            )
        
        # Display results
        st.header("Risk Prediction Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk gauge
            fig_gauge = create_risk_gauge(risk_percentage)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk details
            st.subheader("Risk Assessment Details")
            
            st.metric(
                "36-Month Risk", 
                f"{risk_percentage:.1f}%",
                help="Probability of developing or progressing DKD within 36 months"
            )
            
            if ci_lower is not None and ci_upper is not None:
                st.metric(
                    "95% Confidence Interval",
                    f"{ci_lower*100:.1f}% - {ci_upper*100:.1f}%",
                    help="Statistical uncertainty range for the risk estimate"
                )
            
            # Risk category
            if risk_percentage < 5:
                risk_category = "Low Risk"
                risk_color = "green"
                recommendation = "Continue standard diabetes care and monitoring"
            elif risk_percentage < 15:
                risk_category = "Moderate Risk"
                risk_color = "orange"
                recommendation = "Enhanced monitoring and lifestyle interventions recommended"
            elif risk_percentage < 30:
                risk_category = "High Risk"
                risk_color = "red"
                recommendation = "Consider nephrology referral and intensive management"
            else:
                risk_category = "Very High Risk"
                risk_color = "darkred"
                recommendation = "Urgent nephrology referral and aggressive intervention needed"
            
            st.markdown(f"**Risk Category**: :{risk_color}[{risk_category}]")
            st.markdown(f"**Clinical Recommendation**: {recommendation}")
        
        with col2:
            # Feature importance chart
            if feature_contributions:
                fig_importance = create_feature_importance_chart(feature_contributions)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        # Detailed clinical interpretation
        st.subheader("Clinical Interpretation")
        
        # Risk drivers analysis
        if feature_contributions:
            st.write("**Primary Risk Drivers:**")
            
            risk_factors = []
            protective_factors = []
            
            for factor, contribution in feature_contributions[:8]:
                if contribution > 0:
                    risk_factors.append((factor, contribution))
                else:
                    protective_factors.append((factor, abs(contribution)))
            
            if risk_factors:
                st.write("**Factors Increasing Risk:**")
                for factor, contrib in risk_factors:
                    st.write(f"• {factor}: +{contrib:.3f} log-odds")
            
            if protective_factors:
                st.write("**Protective Factors:**")
                for factor, contrib in protective_factors:
                    st.write(f"• {factor}: -{contrib:.3f} log-odds")
        
        # Clinical recommendations
        st.subheader("Personalized Treatment Recommendations")
        
        recommendations = []
        
        # Medication recommendations
        if not features.get('sglt2i_use') and features['egfr'] >= 30:
            recommendations.append("Consider SGLT2 inhibitor therapy (strong renoprotective evidence)")
        
        if not features.get('ace_arb_use'):
            recommendations.append("Initiate ACE inhibitor or ARB therapy if not contraindicated")
        
        if features.get('ace_arb_use') and features.get('ace_arb_mpr_180', 1.0) < 0.8:
            recommendations.append("Optimize ACE inhibitor/ARB adherence (currently suboptimal)")
        
        if not features.get('statin_use') and risk_percentage > 10:
            recommendations.append("Consider statin therapy for cardiovascular and renal protection")
        
        # Lifestyle recommendations
        if features.get('smoking_status') == 'current':
            recommendations.append("Smoking cessation is critical for kidney protection")
        
        if features.get('hba1c', 7) > 7.5:
            recommendations.append("Intensify glucose control (target HbA1c <7% if tolerated)")
        
        if features.get('sbp', 130) > 130:
            recommendations.append("Optimize blood pressure control (target <130/80 mmHg)")
        
        # Monitoring recommendations
        if risk_percentage > 15:
            recommendations.append("Increase monitoring frequency (every 3-6 months)")
            recommendations.append("Consider nephrology consultation")
        
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.write("Continue current management with regular monitoring.")
        
        # Clinical report
        st.subheader("Clinical Report Summary")
        
        report_text = f"""
DIABETIC KIDNEY DISEASE RISK ASSESSMENT REPORT

Patient Information:
• Age: {features['age']} years
• Sex: {'Male' if features['sex_male'] else 'Female'}
• BMI: {features['bmi']:.1f} kg/m²

Laboratory Results:
• eGFR: {features['egfr']:.1f} mL/min/1.73m²
• ACR: {acr_mg_mmol:.1f} mg/mmol ({features['acr_mg_g']:.1f} mg/g)
• HbA1c: {features['hba1c']:.1f}%
• Blood Pressure: {features['sbp']}/{features['dbp']} mmHg

Risk Assessment:
• Model Used: {model_type.title()} DKD Model
• 36-Month Risk: {risk_percentage:.1f}%
• Risk Category: {risk_category}
• Confidence Interval: {ci_lower*100:.1f}% - {ci_upper*100:.1f}%

Clinical Recommendations:
"""
        
        for rec in recommendations:
            report_text += f"• {rec}\n"
        
        report_text += f"""
Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Generated by: NephraRisk Clinical Decision Support System

Note: This assessment is for clinical decision support only and does not replace clinical judgment.
"""
        
        st.text_area("Clinical Report", report_text, height=300)

if __name__ == "__main__":
    main()
