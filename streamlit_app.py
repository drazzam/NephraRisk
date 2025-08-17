import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math

# Configure Streamlit page
st.set_page_config(
    page_title="NephraRisk - DKD Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified and Accurate Model Coefficients
class DKDModelCoefficients:
    def __init__(self):
        # Model parameters - conservative and realistic
        self.incident_alpha = -4.8
        self.progression_alpha = -4.2
        
        # Core clinical coefficients (realistic values)
        self.coefficients = {
            # Demographics (small effects)
            'age_per_year': 0.020,
            'male_sex': 0.15,
            
            # Core lab values (main drivers)
            'egfr_per_10ml_decline': 0.12,  # Per 10 mL/min decline from 100
            'acr_log_per_unit': 0.30,  # Per log unit of ACR
            'hba1c_per_percent': 0.10,  # Per 1% HbA1c
            
            # Blood pressure
            'sbp_per_10mmhg': 0.03,
            'hypertension_flag': 0.08,  # SBP >140
            
            # Diabetes complications
            'retinopathy_any': 0.25,
            'retinopathy_severe': 0.40,  # Additional for severe/PDR
            'neuropathy': 0.18,
            'cardiovascular_disease': 0.22,
            
            # Protective medications
            'sglt2i_use': -0.40,
            'ace_arb_use': -0.25,
            'statin_use': -0.15,
            
            # Risk medications/markers
            'insulin_use': 0.18,
            
            # Lifestyle factors
            'current_smoking': 0.25,
            'family_history_ckd': 0.20,
            'depression': 0.12,
        }

def convert_acr_units(acr_mg_mmol):
    """Convert ACR from mg/mmol to mg/g"""
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
    """Determine model type internally"""
    if egfr >= 60 and acr_mg_g < 30:
        return "incident"
    else:
        return "progression"

def calculate_risk_score(features, coefficients):
    """Calculate risk score with proper feature tracking"""
    
    # Initialize
    risk_score = 0.0
    active_features = {}  # Track what features are actually contributing
    
    # Demographics
    age = features.get('age', 50)
    if age and age > 18:
        age_contribution = coefficients.coefficients['age_per_year'] * age
        risk_score += age_contribution
        if age > 65:  # Only show as factor if significantly contributing
            active_features['Advanced Age'] = {'effect': 'risk', 'value': age_contribution}
    
    if features.get('sex_male'):
        male_contribution = coefficients.coefficients['male_sex']
        risk_score += male_contribution
        active_features['Male Sex'] = {'effect': 'risk', 'value': male_contribution}
    
    # Core lab values - main risk drivers
    egfr = features.get('egfr', 90)
    if egfr and egfr < 100:
        egfr_decline = (100 - egfr) / 10.0  # How much below 100, per 10 mL
        egfr_contribution = coefficients.coefficients['egfr_per_10ml_decline'] * egfr_decline
        risk_score += egfr_contribution
        
        if egfr < 60:
            active_features['Reduced Kidney Function'] = {'effect': 'risk', 'value': egfr_contribution}
    
    # ACR - major risk factor
    acr_mg_g = features.get('acr_mg_g', 10)
    if acr_mg_g and acr_mg_g > 1:
        acr_log = math.log(acr_mg_g + 1)
        acr_contribution = coefficients.coefficients['acr_log_per_unit'] * acr_log
        risk_score += acr_contribution
        
        if acr_mg_g >= 30:
            if acr_mg_g >= 300:
                active_features['Severe Proteinuria'] = {'effect': 'risk', 'value': acr_contribution}
            else:
                active_features['Mild Proteinuria'] = {'effect': 'risk', 'value': acr_contribution}
    
    # HbA1c
    hba1c = features.get('hba1c', 7)
    if hba1c and hba1c > 6:
        hba1c_excess = hba1c - 6.0  # Above optimal
        hba1c_contribution = coefficients.coefficients['hba1c_per_percent'] * hba1c_excess
        risk_score += hba1c_contribution
        
        if hba1c > 8:
            active_features['Poor Glucose Control'] = {'effect': 'risk', 'value': hba1c_contribution}
    
    # Blood pressure
    sbp = features.get('sbp', 120)
    if sbp and sbp > 120:
        sbp_excess = (sbp - 120) / 10.0
        sbp_contribution = coefficients.coefficients['sbp_per_10mmhg'] * sbp_excess
        risk_score += sbp_contribution
        
        if sbp > 140:
            hypertension_contribution = coefficients.coefficients['hypertension_flag']
            risk_score += hypertension_contribution
            active_features['High Blood Pressure'] = {'effect': 'risk', 'value': sbp_contribution + hypertension_contribution}
    
    # Diabetes complications - only if present
    if features.get('retinopathy'):
        retinopathy_contribution = coefficients.coefficients['retinopathy_any']
        risk_score += retinopathy_contribution
        
        # Check severity
        severity = features.get('retinopathy_severity', 'mild_npdr')
        if severity in ['severe_npdr', 'pdr']:
            severe_contribution = coefficients.coefficients['retinopathy_severe']
            risk_score += severe_contribution
            active_features['Advanced Eye Disease'] = {'effect': 'risk', 'value': retinopathy_contribution + severe_contribution}
        else:
            active_features['Diabetic Eye Disease'] = {'effect': 'risk', 'value': retinopathy_contribution}
    
    if features.get('neuropathy_dx'):
        neuropathy_contribution = coefficients.coefficients['neuropathy']
        risk_score += neuropathy_contribution
        active_features['Diabetic Neuropathy'] = {'effect': 'risk', 'value': neuropathy_contribution}
    
    if features.get('ascvd_dx'):
        cvd_contribution = coefficients.coefficients['cardiovascular_disease']
        risk_score += cvd_contribution
        active_features['Cardiovascular Disease'] = {'effect': 'risk', 'value': cvd_contribution}
    
    # Protective medications - only if present
    if features.get('sglt2i_use'):
        sglt2i_contribution = coefficients.coefficients['sglt2i_use']
        risk_score += sglt2i_contribution  # Negative value
        active_features['SGLT2 Inhibitor'] = {'effect': 'protective', 'value': abs(sglt2i_contribution)}
    
    if features.get('ace_arb_use'):
        ace_arb_contribution = coefficients.coefficients['ace_arb_use']
        risk_score += ace_arb_contribution  # Negative value
        active_features['ACE Inhibitor/ARB'] = {'effect': 'protective', 'value': abs(ace_arb_contribution)}
    
    if features.get('statin_use'):
        statin_contribution = coefficients.coefficients['statin_use']
        risk_score += statin_contribution  # Negative value
        active_features['Statin Therapy'] = {'effect': 'protective', 'value': abs(statin_contribution)}
    
    # Risk factors - only if present
    if features.get('insulin_used'):
        insulin_contribution = coefficients.coefficients['insulin_use']
        risk_score += insulin_contribution
        active_features['Insulin Therapy'] = {'effect': 'risk', 'value': insulin_contribution}
    
    if features.get('smoking_status') == 'current':
        smoking_contribution = coefficients.coefficients['current_smoking']
        risk_score += smoking_contribution
        active_features['Current Smoking'] = {'effect': 'risk', 'value': smoking_contribution}
    
    if features.get('family_hx_ckd'):
        family_hx_contribution = coefficients.coefficients['family_history_ckd']
        risk_score += family_hx_contribution
        active_features['Family History of Kidney Disease'] = {'effect': 'risk', 'value': family_hx_contribution}
    
    if features.get('depression_dx'):
        depression_contribution = coefficients.coefficients['depression']
        risk_score += depression_contribution
        active_features['Depression'] = {'effect': 'risk', 'value': depression_contribution}
    
    return risk_score, active_features

def calculate_final_risk(risk_score, model_type, coefficients):
    """Convert risk score to final probability"""
    
    # Add model-specific intercept
    if model_type == "incident":
        total_score = coefficients.incident_alpha + risk_score
    else:
        total_score = coefficients.progression_alpha + risk_score
    
    # Convert to monthly probability
    monthly_prob = 1 / (1 + math.exp(-total_score))
    
    # Cap monthly probability
    monthly_prob = min(monthly_prob, 0.06)  # Max 6% per month
    
    # Calculate 36-month cumulative risk
    survival_prob = (1 - monthly_prob) ** 36
    cumulative_risk = 1 - survival_prob
    
    # Apply final calibration to realistic range
    if model_type == "incident":
        # More conservative for incident
        final_risk = cumulative_risk * 0.5 + 0.02
    else:
        # Less conservative for progression
        final_risk = cumulative_risk * 0.7 + 0.05
    
    # Final bounds
    final_risk = max(0.01, min(0.80, final_risk))
    
    return final_risk * 100  # Convert to percentage

def create_risk_gauge(risk_percentage):
    """Create risk gauge"""
    
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

def create_risk_factors_chart(active_features):
    """Create risk factors chart from actual contributing features"""
    
    if not active_features:
        return None
    
    # Sort by contribution value
    sorted_features = sorted(active_features.items(), key=lambda x: x[1]['value'], reverse=True)
    
    # Take top 8 features
    top_features = sorted_features[:8]
    
    factors = [item[0] for item in top_features]
    effects = [item[1]['effect'] for item in top_features]
    values = [item[1]['value'] if item[1]['effect'] == 'risk' else -item[1]['value'] for item in top_features]
    colors = ['red' if effect == 'risk' else 'green' for effect in effects]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=factors,
        orientation='h',
        marker_color=colors,
        showlegend=False
    ))
    
    fig.update_layout(
        title="Risk Factors",
        xaxis_title="Contribution",
        yaxis_title="",
        height=max(300, len(factors) * 35),
        yaxis={'categoryorder': 'total ascending'},
        xaxis={'showticklabels': False}
    )
    
    return fig

# Main Application
def main():
    st.title("NephraRisk - Diabetic Kidney Disease and Diabetic Nephropathy Risk Prediction")
    
    # Initialize model
    coefficients = DKDModelCoefficients()
    
    # Initialize session state
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
    
    # Patient Information Entry
    st.header("Patient Information Entry")
    
    # Demographics
    st.subheader("Patient Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55)
        st.session_state.patient_data['age'] = age
        
    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        st.session_state.patient_data['sex_male'] = sex == "Male"
        
    with col3:
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        st.session_state.patient_data['bmi'] = bmi
    
    # Laboratory Values
    st.subheader("Laboratory Values")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        egfr = st.number_input("Estimated GFR (mL/min/1.73m²)", min_value=5.0, max_value=150.0, value=90.0, step=1.0)
        st.session_state.patient_data['egfr'] = egfr
        
    with col2:
        acr_mg_mmol = st.number_input("ACR (mg/mmol)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
        acr_mg_g = convert_acr_units(acr_mg_mmol)
        st.session_state.patient_data['acr_mg_g'] = acr_mg_g
        st.info(f"ACR converted: {acr_mg_g:.1f} mg/g ({get_acr_category(acr_mg_g)})")
        
    with col3:
        hba1c = st.number_input("HbA1c (%)", min_value=5.0, max_value=15.0, value=7.0, step=0.1)
        st.session_state.patient_data['hba1c'] = hba1c
    
    # Blood Pressure
    st.subheader("Blood Pressure")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=125)
        st.session_state.patient_data['sbp'] = sbp
        
    with col2:
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)
        st.session_state.patient_data['dbp'] = dbp
        
    with col3:
        pulse_pressure = sbp - dbp
        st.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
    
    # Diabetes Complications
    st.subheader("Diabetes Complications")
    
    col1, col2 = st.columns(2)
    with col1:
        retinopathy = st.selectbox("Diabetic Retinopathy", ["No", "Yes"])
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        neuropathy = st.selectbox("Diabetic Neuropathy", ["No", "Yes"])
        st.session_state.patient_data['neuropathy_dx'] = neuropathy == "Yes"
        
    with col2:
        ascvd = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
        st.session_state.patient_data['ascvd_dx'] = ascvd == "Yes"
        
    with col3:
        anemia = st.selectbox("Anemia", ["No", "Yes"])
        st.session_state.patient_data['anemia_dx'] = anemia == "Yes"
    
    # Medications
    st.subheader("Current Medications")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sglt2i_use = st.selectbox("SGLT2 Inhibitor", ["No", "Yes"])
        st.session_state.patient_data['sglt2i_use'] = sglt2i_use == "Yes"
        
    with col2:
        ace_arb_use = st.selectbox("ACE Inhibitor/ARB", ["No", "Yes"])
        st.session_state.patient_data['ace_arb_use'] = ace_arb_use == "Yes"
        
    with col3:
        statin_use = st.selectbox("Statin", ["No", "Yes"])
        st.session_state.patient_data['statin_use'] = statin_use == "Yes"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        insulin_used = st.selectbox("Insulin Therapy", ["No", "Yes"])
        st.session_state.patient_data['insulin_used'] = insulin_used == "Yes"
        
    with col2:
        glp1_use = st.selectbox("GLP-1 Agonist", ["No", "Yes"])
        st.session_state.patient_data['glp1_use'] = glp1_use == "Yes"
        
    with col3:
        mra_use = st.selectbox("MRA", ["No", "Yes"])
        st.session_state.patient_data['mra_use'] = mra_use == "Yes"
    
    # Lifestyle and History
    st.subheader("Lifestyle and Medical History")
    
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
            pack_years = st.number_input("Pack-Years", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
            st.session_state.patient_data['pack_years'] = pack_years
    
    col1, col2, col3 = st.columns(3)
    with col1:
        depression = st.selectbox("Depression", ["No", "Yes"])
        st.session_state.patient_data['depression_dx'] = depression == "Yes"
        
    with col2:
        family_hx = st.selectbox("Family History of Kidney Disease", ["No", "Yes"])
        st.session_state.patient_data['family_hx_ckd'] = family_hx == "Yes"
        
    with col3:
        nsaid_use = st.selectbox("Regular NSAID Use", ["No", "Yes"])
        st.session_state.patient_data['nsaid_use'] = nsaid_use == "Yes"
    
    # Calculate Risk Button
    st.markdown("---")
    if st.button("Calculate Risk", type="primary"):
        
        features = st.session_state.patient_data
        
        # Determine model type
        model_type = determine_model_type(features['egfr'], features['acr_mg_g'])
        
        with st.spinner("Calculating risk..."):
            # Calculate risk
            risk_score, active_features = calculate_risk_score(features, coefficients)
            risk_percentage = calculate_final_risk(risk_score, model_type, coefficients)
        
        # Display Results
        st.header("Risk Prediction Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk gauge
            fig_gauge = create_risk_gauge(risk_percentage)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.metric("36-Month Risk", f"{risk_percentage:.1f}%")
        
        with col2:
            # Risk factors chart
            if active_features:
                fig_factors = create_risk_factors_chart(active_features)
                if fig_factors:
                    st.plotly_chart(fig_factors, use_container_width=True)
        
        # Clinical Interpretation
        st.subheader("Clinical Interpretation")
        
        if active_features:
            # Separate risk and protective factors
            risk_factors = [name for name, info in active_features.items() if info['effect'] == 'risk']
            protective_factors = [name for name, info in active_features.items() if info['effect'] == 'protective']
            
            if risk_factors:
                st.write("**Factors Increasing Risk:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            
            if protective_factors:
                st.write("**Protective Factors:**")
                for factor in protective_factors:
                    st.write(f"• {factor}")
        else:
            st.write("• Patient has baseline risk profile with no major risk factors or protective treatments identified.")

if __name__ == "__main__":
    main()
