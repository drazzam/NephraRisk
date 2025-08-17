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

# Enhanced Model Coefficients with Clinically Sensitive Predictions
class DKDModelCoefficients:
    def __init__(self):
        # Incident DKD Model Parameters (calibrated for sensitivity)
        self.incident_alpha = -5.2  # Conservative baseline
        self.incident_platt_a = -1.8
        self.incident_platt_b = 0.9
        
        # Progression Model Parameters (higher baseline risk)
        self.progression_alpha = -4.1
        self.progression_platt_a = -1.5
        self.progression_platt_b = 0.95
        
        # Core Clinical Variables (Clinically Sensitive)
        self.core_coefficients = {
            # Age effects (progressive with age)
            'age_per_year': 0.025,  # Significant age effect
            'age_over_65_bonus': 0.15,  # Additional risk >65
            
            # Sex effects
            'sex_male': 0.18,  # Male disadvantage
            
            # BMI effects (non-linear)
            'bmi_per_unit': 0.025,  # Base BMI effect
            'obesity_bonus': 0.12,  # Additional if BMI >30
            
            # HbA1c effects (progressive with poor control)
            'hba1c_per_percent': 0.12,  # Base effect
            'hba1c_poor_control_bonus': 0.08,  # Additional if >9%
            
            # Blood pressure (sensitive to hypertension)
            'sbp_per_10mmhg': 0.06,
            'hypertension_bonus': 0.10,  # Additional if SBP >140
            'dbp_per_10mmhg': 0.04,
            
            # eGFR effects (highly sensitive to kidney function)
            'egfr_normal': 0.0,  # Reference eGFR >90
            'egfr_mildly_reduced': 0.08,  # eGFR 60-89
            'egfr_moderately_reduced': 0.25,  # eGFR 30-59
            'egfr_severely_reduced': 0.55,  # eGFR 15-29
            'egfr_very_severe': 0.85,  # eGFR <15
            
            # ACR effects (highly sensitive to proteinuria)
            'acr_normal': 0.0,  # Reference ACR <30 mg/g
            'acr_microalbuminuria': 0.35,  # 30-300 mg/g
            'acr_macroalbuminuria': 0.70,  # >300 mg/g
            'acr_severe': 1.0,  # >1000 mg/g
        }
        
        # Diabetes Complications (Strong clinical effects)
        self.complications_coefficients = {
            # Retinopathy (progressive severity)
            'retinopathy_base': 0.20,
            'retinopathy_mild': 0.15,  # Additional for mild
            'retinopathy_moderate': 0.35,  # Additional for moderate
            'retinopathy_severe': 0.55,  # Additional for severe
            'retinopathy_pdr': 0.75,  # Additional for PDR
            'macular_edema_bonus': 0.25,
            
            # Other complications
            'neuropathy_dx': 0.22,
            'ascvd_dx': 0.28,  # Strong CV-renal connection
            'foot_ulcer_dx': 0.35,  # Advanced diabetes marker
            'anemia_dx': 0.18,
        }
        
        # Medications (Evidence-based protective effects)
        self.medications_coefficients = {
            # SGLT2 inhibitors (strong protection)
            'sglt2i_base': -0.45,  # Strong base protection
            'sglt2i_good_adherence': -0.20,  # Additional if adherence >80%
            'sglt2i_excellent_adherence': -0.35,  # Additional if adherence >90%
            
            # ACE/ARB (dose and adherence sensitive)
            'ace_arb_base': -0.25,
            'ace_arb_good_adherence': -0.15,  # Additional if adherence >80%
            'ace_arb_optimal_dose': -0.18,  # Additional if dose Z-score >0
            
            # Statins (protective)
            'statin_base': -0.18,
            'statin_good_adherence': -0.10,  # Additional if adherence >80%
            
            # Other medications
            'insulin_baseline': 0.15,  # Disease severity marker
            'insulin_duration_bonus': 0.05,  # Per 5 years
            'glp1_base': -0.12,  # Protective
            'mra_base': -0.22,  # Strong protection
        }
        
        # Lifestyle Factors (Clinically Significant)
        self.lifestyle_coefficients = {
            # Smoking (dose-dependent)
            'smoking_current': 0.35,
            'smoking_former': 0.12,
            'heavy_smoking_bonus': 0.18,  # >20 pack-years
            'recent_quit_bonus': -0.08,  # <2 years quit
            
            # Social determinants
            'depression_dx': 0.16,
            'family_hx_ckd': 0.22,  # Strong genetic component
            'nsaid_chronic': 0.14,
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
    """Determine whether to use incident or progression model (internal only)"""
    if egfr >= 60 and acr_mg_g < 30:
        return "incident"
    else:
        return "progression"

def calculate_enhanced_monthly_hazard(features, coefficients, alpha, model_type):
    """Calculate monthly hazard using clinically sensitive coefficients"""
    
    # Start with intercept
    linear_predictor = alpha
    
    # Age effects (progressive)
    age = features.get('age', 50)
    linear_predictor += coefficients.core_coefficients['age_per_year'] * age
    if age > 65:
        linear_predictor += coefficients.core_coefficients['age_over_65_bonus']
    
    # Sex effects
    if features.get('sex_male'):
        linear_predictor += coefficients.core_coefficients['sex_male']
    
    # BMI effects (non-linear)
    bmi = features.get('bmi', 25)
    linear_predictor += coefficients.core_coefficients['bmi_per_unit'] * bmi
    if bmi > 30:
        linear_predictor += coefficients.core_coefficients['obesity_bonus']  # Fixed reference
    
    # HbA1c effects (progressive with poor control)
    hba1c = features.get('hba1c', 7)
    linear_predictor += coefficients.core_coefficients['hba1c_per_percent'] * hba1c
    if hba1c > 9:
        linear_predictor += coefficients.core_coefficients['hba1c_poor_control_bonus']
    
    # Blood pressure (sensitive to hypertension)
    sbp = features.get('sbp', 120)
    dbp = features.get('dbp', 80)
    linear_predictor += coefficients.core_coefficients['sbp_per_10mmhg'] * (sbp / 10.0)
    linear_predictor += coefficients.core_coefficients['dbp_per_10mmhg'] * (dbp / 10.0)
    if sbp > 140:
        linear_predictor += coefficients.core_coefficients['hypertension_bonus']
    
    # eGFR effects (categorical and highly sensitive)
    egfr = features.get('egfr', 90)
    if egfr >= 90:
        egfr_effect = coefficients.core_coefficients['egfr_normal']
    elif egfr >= 60:
        egfr_effect = coefficients.core_coefficients['egfr_mildly_reduced']
    elif egfr >= 30:
        egfr_effect = coefficients.core_coefficients['egfr_moderately_reduced']
    elif egfr >= 15:
        egfr_effect = coefficients.core_coefficients['egfr_severely_reduced']
    else:
        egfr_effect = coefficients.core_coefficients['egfr_very_severe']
    linear_predictor += egfr_effect
    
    # ACR effects (categorical and highly sensitive)
    acr_mg_g = features.get('acr_mg_g', 15)
    if acr_mg_g < 30:
        acr_effect = coefficients.core_coefficients['acr_normal']
    elif acr_mg_g < 300:
        acr_effect = coefficients.core_coefficients['acr_microalbuminuria']
    elif acr_mg_g < 1000:
        acr_effect = coefficients.core_coefficients['acr_macroalbuminuria']
    else:
        acr_effect = coefficients.core_coefficients['acr_severe']
    linear_predictor += acr_effect
    
    # Retinopathy effects (progressive severity)
    if features.get('retinopathy'):
        linear_predictor += coefficients.complications_coefficients['retinopathy_base']
        
        severity = features.get('retinopathy_severity', 'mild_npdr')
        if severity == 'mild_npdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_mild']
        elif severity == 'moderate_npdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_moderate']
        elif severity == 'severe_npdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_severe']
        elif severity == 'pdr':
            linear_predictor += coefficients.complications_coefficients['retinopathy_pdr']
        
        if features.get('macular_edema'):
            linear_predictor += coefficients.complications_coefficients['macular_edema_bonus']
    
    # Other complications
    if features.get('neuropathy_dx'):
        linear_predictor += coefficients.complications_coefficients['neuropathy_dx']
    
    if features.get('ascvd_dx'):
        linear_predictor += coefficients.complications_coefficients['ascvd_dx']
    
    if features.get('anemia_dx'):
        linear_predictor += coefficients.complications_coefficients['anemia_dx']
    
    # SGLT2 inhibitor effects (adherence-sensitive)
    if features.get('sglt2i_use'):
        linear_predictor += coefficients.medications_coefficients['sglt2i_base']
        
        pdc = features.get('sglt2i_pdc_180', 0.8)
        if pdc > 0.9:
            linear_predictor += coefficients.medications_coefficients['sglt2i_excellent_adherence']
        elif pdc > 0.8:
            linear_predictor += coefficients.medications_coefficients['sglt2i_good_adherence']
    
    # ACE/ARB effects (adherence and dose sensitive)
    if features.get('ace_arb_use'):
        linear_predictor += coefficients.medications_coefficients['ace_arb_base']
        
        mpr = features.get('ace_arb_mpr_180', 0.8)
        if mpr > 0.8:
            linear_predictor += coefficients.medications_coefficients['ace_arb_good_adherence']
        
        dose_std = features.get('ace_arb_dose_std', 0)
        if dose_std > 0:
            linear_predictor += coefficients.medications_coefficients['ace_arb_optimal_dose']
    
    # Statin effects
    if features.get('statin_use'):
        linear_predictor += coefficients.medications_coefficients['statin_base']
        
        statin_mpr = features.get('statin_mpr_180', 0.75)
        if statin_mpr > 0.8:
            linear_predictor += coefficients.medications_coefficients['statin_good_adherence']
    
    # Insulin (disease severity marker)
    if features.get('insulin_used'):
        linear_predictor += coefficients.medications_coefficients['insulin_baseline']
    
    # GLP-1 protective
    if features.get('glp1_use'):
        linear_predictor += coefficients.medications_coefficients['glp1_base']
    
    # MRA protective
    if features.get('mra_use'):
        linear_predictor += coefficients.medications_coefficients['mra_base']
    
    # Smoking effects
    smoking_status = features.get('smoking_status', 'never')
    if smoking_status == 'current':
        linear_predictor += coefficients.lifestyle_coefficients['smoking_current']
    elif smoking_status == 'former':
        linear_predictor += coefficients.lifestyle_coefficients['smoking_former']
    
    # Heavy smoking bonus
    pack_years = features.get('pack_years_per10', 0) * 10
    if pack_years > 20:
        linear_predictor += coefficients.lifestyle_coefficients['heavy_smoking_bonus']
    
    # Social factors
    if features.get('depression_dx'):
        linear_predictor += coefficients.lifestyle_coefficients['depression_dx']
    
    if features.get('family_hx_ckd'):
        linear_predictor += coefficients.lifestyle_coefficients['family_hx_ckd']
    
    if features.get('nsaid_chronic_use'):
        linear_predictor += coefficients.lifestyle_coefficients['nsaid_chronic']
    
    # Convert to monthly probability
    monthly_hazard = 1 / (1 + np.exp(-linear_predictor))
    
    # Cap to realistic monthly rates
    monthly_hazard = min(monthly_hazard, 0.08)  # Max 8% per month
    
    return monthly_hazard, linear_predictor

def calculate_36_month_risk(monthly_hazard):
    """Calculate 36-month cumulative risk with realistic modeling"""
    
    # Apply slight decay in hazard over time (clinical reality)
    survival_prob = 1.0
    current_hazard = monthly_hazard
    
    for month in range(36):
        # Slight hazard decay over time (intervention effects)
        if month > 12:
            decay_factor = 0.995  # 0.5% decay per month after year 1
            current_hazard = max(current_hazard * decay_factor, monthly_hazard * 0.7)
        
        survival_prob *= (1 - current_hazard)
    
    cumulative_risk = 1 - survival_prob
    return cumulative_risk

def apply_enhanced_calibration(raw_risk, model_type, coefficients):
    """Apply enhanced calibration for realistic predictions"""
    
    # Ensure reasonable bounds
    raw_risk = max(0.001, min(0.95, raw_risk))
    
    # Apply different calibration based on model type
    if model_type == "incident":
        # More conservative calibration for incident model
        calibrated_risk = raw_risk * 0.6 + 0.02  # Scale down with small baseline
    else:
        # Less conservative for progression model
        calibrated_risk = raw_risk * 0.8 + 0.05  # Higher baseline for progression
    
    # Final bounds check
    calibrated_risk = max(0.01, min(0.85, calibrated_risk))
    
    return calibrated_risk

def calculate_feature_importance_simple(features, coefficients, model_type):
    """Calculate simplified feature importance for display"""
    
    contributions = {}
    
    # Core risk factors
    age = features.get('age', 50)
    if age > 65:
        contributions['Advanced Age (>65 years)'] = 'risk'
    
    if features.get('sex_male'):
        contributions['Male Sex'] = 'risk'
    
    # BMI check
    bmi = features.get('bmi', 25)
    if bmi > 30:
        contributions['Obesity (BMI >30)'] = 'risk'
    
    hba1c = features.get('hba1c', 7)
    if hba1c > 8:
        contributions['Poor Glucose Control (HbA1c >8%)'] = 'risk'
    elif hba1c > 9:
        contributions['Very Poor Glucose Control (HbA1c >9%)'] = 'risk'
    
    egfr = features.get('egfr', 90)
    if egfr < 60:
        if egfr < 30:
            contributions['Severely Reduced Kidney Function (eGFR <30)'] = 'risk'
        else:
            contributions['Reduced Kidney Function (eGFR <60)'] = 'risk'
    
    acr_mg_g = features.get('acr_mg_g', 15)
    if acr_mg_g >= 30:
        if acr_mg_g >= 300:
            contributions['Severe Proteinuria (Macroalbuminuria)'] = 'risk'
        else:
            contributions['Mild Proteinuria (Microalbuminuria)'] = 'risk'
    
    sbp = features.get('sbp', 120)
    if sbp > 140:
        contributions['High Blood Pressure'] = 'risk'
    
    # Complications
    if features.get('retinopathy'):
        severity = features.get('retinopathy_severity', 'mild_npdr')
        if severity in ['severe_npdr', 'pdr']:
            contributions['Advanced Diabetic Eye Disease'] = 'risk'
        else:
            contributions['Diabetic Eye Disease'] = 'risk'
    
    if features.get('neuropathy_dx'):
        contributions['Diabetic Nerve Disease'] = 'risk'
    
    if features.get('ascvd_dx'):
        contributions['Cardiovascular Disease'] = 'risk'
    
    # Protective medications
    if features.get('sglt2i_use'):
        pdc = features.get('sglt2i_pdc_180', 0.8)
        if pdc > 0.8:
            contributions['SGLT2 Inhibitor (Good Adherence)'] = 'protective'
        else:
            contributions['SGLT2 Inhibitor (Poor Adherence)'] = 'risk'
    
    if features.get('ace_arb_use'):
        mpr = features.get('ace_arb_mpr_180', 0.8)
        if mpr > 0.8:
            contributions['ACE Inhibitor/ARB (Good Adherence)'] = 'protective'
        else:
            contributions['ACE Inhibitor/ARB (Poor Adherence)'] = 'risk'
    
    if features.get('statin_use'):
        contributions['Statin Therapy'] = 'protective'
    
    if features.get('mra_use'):
        contributions['MRA Therapy'] = 'protective'
    
    if features.get('glp1_use'):
        contributions['GLP-1 Agonist'] = 'protective'
    
    # Lifestyle factors
    smoking_status = features.get('smoking_status', 'never')
    if smoking_status == 'current':
        contributions['Current Smoking'] = 'risk'
    
    if features.get('family_hx_ckd'):
        contributions['Family History of Kidney Disease'] = 'risk'
    
    if features.get('depression_dx'):
        contributions['Depression'] = 'risk'
    
    if features.get('nsaid_chronic_use'):
        contributions['Regular NSAID Use'] = 'risk'
    
    return contributions

def create_risk_gauge(risk_percentage):
    """Create a simplified risk gauge chart"""
    
    # Determine color based on risk
    if risk_percentage < 5:
        color = "green"
    elif risk_percentage < 15:
        color = "yellow"
    elif risk_percentage < 30:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "36-Month Risk Prediction"},
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

def create_simple_risk_factors_chart(contributions):
    """Create simplified risk factors visualization"""
    
    if not contributions:
        return None
    
    # Separate risk and protective factors
    risk_factors = [factor for factor, effect in contributions.items() if effect == 'risk']
    protective_factors = [factor for factor, effect in contributions.items() if effect == 'protective']
    
    # Create data for chart
    all_factors = risk_factors + protective_factors
    values = [1] * len(risk_factors) + [-1] * len(protective_factors)
    colors = ['red'] * len(risk_factors) + ['green'] * len(protective_factors)
    
    if not all_factors:
        return None
    
    fig = go.Figure(go.Bar(
        x=values,
        y=all_factors,
        orientation='h',
        marker_color=colors,
        showlegend=False
    ))
    
    fig.update_layout(
        title="Risk Factors",
        xaxis_title="Effect",
        yaxis_title="",
        height=max(300, len(all_factors) * 30),
        yaxis={'categoryorder': 'total ascending'},
        xaxis={'showticklabels': False}
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
        st.session_state.patient_data['acr_mg_mmol'] = acr_mg_mmol
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
        
        # Determine model type (internal only)
        model_type = determine_model_type(features['egfr'], features['acr_mg_g'])
        
        # Calculate prediction
        with st.spinner("Calculating risk prediction..."):
            
            # Get appropriate parameters
            if model_type == "incident":
                alpha = coefficients.incident_alpha
            else:
                alpha = coefficients.progression_alpha
            
            # Calculate enhanced monthly hazard and risk
            monthly_hazard, linear_predictor = calculate_enhanced_monthly_hazard(
                features, coefficients, alpha, model_type
            )
            
            raw_risk = calculate_36_month_risk(monthly_hazard)
            calibrated_risk = apply_enhanced_calibration(raw_risk, model_type, coefficients)
            
            risk_percentage = calibrated_risk * 100
            
            # Calculate simplified feature importance
            feature_contributions = calculate_feature_importance_simple(
                features, coefficients, model_type
            )
        
        # Display results
        st.header("Risk Prediction Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk gauge
            fig_gauge = create_risk_gauge(risk_percentage)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.metric(
                "36-Month Risk", 
                f"{risk_percentage:.1f}%",
                help="Probability of developing or progressing DKD within 36 months"
            )
        
        with col2:
            # Feature importance chart
            if feature_contributions:
                fig_importance = create_simple_risk_factors_chart(feature_contributions)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        # Clinical interpretation
        st.subheader("Clinical Interpretation")
        
        if feature_contributions:
            # Separate risk and protective factors
            risk_factors = [factor for factor, effect in feature_contributions.items() if effect == 'risk']
            protective_factors = [factor for factor, effect in feature_contributions.items() if effect == 'protective']
            
            if risk_factors:
                st.write("**Factors Increasing Risk:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            
            if protective_factors:
                st.write("**Protective Factors:**")
                for factor in protective_factors:
                    st.write(f"• {factor}")
                    
            if not risk_factors and not protective_factors:
                st.write("• Patient profile indicates average risk based on age and baseline clinical parameters")

if __name__ == "__main__":
    main()
