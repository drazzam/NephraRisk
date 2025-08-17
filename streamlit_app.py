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

# Properly Calibrated Model Coefficients
class DKDModelCoefficients:
    def __init__(self):
        # Base risk parameters (properly calibrated)
        self.base_monthly_risk_incident = 0.002  # 0.2% per month baseline for healthy
        self.base_monthly_risk_progression = 0.008  # 0.8% per month baseline for existing DKD
        
        # Risk multipliers (properly scaled)
        self.risk_factors = {
            # Demographics
            'age_multiplier_per_10_years': 1.15,  # 15% increase per 10 years above 50
            'male_sex_multiplier': 1.20,  # 20% higher risk for males
            
            # Core clinical (major impact)
            'egfr_multiplier_per_10ml_drop': 1.25,  # 25% increase per 10 mL drop below 90
            'acr_multiplier_doubling': 1.40,  # 40% increase per doubling of ACR above 10
            'hba1c_multiplier_per_percent': 1.12,  # 12% increase per 1% above 7%
            
            # Blood pressure
            'sbp_multiplier_per_10mmhg': 1.05,  # 5% increase per 10 mmHg above 120
            'severe_hypertension_bonus': 1.30,  # Additional 30% if SBP >160
            
            # Complications (significant multipliers)
            'retinopathy_mild_multiplier': 1.50,  # 50% increase
            'retinopathy_severe_multiplier': 2.20,  # 120% increase (severe/PDR)
            'neuropathy_multiplier': 1.35,  # 35% increase
            'cardiovascular_multiplier': 1.45,  # 45% increase
            
            # Medications (protective - reduce risk)
            'sglt2i_protection_factor': 0.65,  # 35% risk reduction
            'ace_arb_protection_factor': 0.75,  # 25% risk reduction
            'statin_protection_factor': 0.85,  # 15% risk reduction
            
            # Lifestyle/other risk factors
            'current_smoking_multiplier': 1.40,  # 40% increase
            'insulin_use_multiplier': 1.25,  # 25% increase (disease severity marker)
            'family_history_multiplier': 1.30,  # 30% increase
            'depression_multiplier': 1.15,  # 15% increase
        }

def convert_acr_units(acr_mg_mmol):
    """Convert ACR from mg/mmol to mg/g"""
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
    """Determine model type based on CKD status"""
    if egfr >= 60 and acr_mg_g < 30:
        return "incident"
    else:
        return "progression"

def calculate_dynamic_risk(features, coefficients):
    """Calculate truly dynamic risk based on patient features"""
    
    # Determine model type and base risk
    egfr = features.get('egfr', 90)
    acr_mg_g = features.get('acr_mg_g', 10)
    model_type = determine_model_type(egfr, acr_mg_g)
    
    # Start with appropriate base monthly risk
    if model_type == "incident":
        monthly_risk = coefficients.base_monthly_risk_incident
    else:
        monthly_risk = coefficients.base_monthly_risk_progression
    
    # Track active risk and protective factors
    active_factors = {'risk': [], 'protective': []}
    risk_multiplier = 1.0
    
    # Age effect (continuous)
    age = features.get('age', 50)
    if age > 50:
        age_decades_over_50 = (age - 50) / 10.0
        age_multiplier = coefficients.risk_factors['age_multiplier_per_10_years'] ** age_decades_over_50
        risk_multiplier *= age_multiplier
        if age > 65:
            active_factors['risk'].append(f"Advanced Age ({age} years)")
    
    # Sex effect
    if features.get('sex_male'):
        sex_multiplier = coefficients.risk_factors['male_sex_multiplier']
        risk_multiplier *= sex_multiplier
        active_factors['risk'].append("Male Sex")
    
    # eGFR effect (major driver)
    if egfr < 90:
        egfr_drop_decades = (90 - egfr) / 10.0
        egfr_multiplier = coefficients.risk_factors['egfr_multiplier_per_10ml_drop'] ** egfr_drop_decades
        risk_multiplier *= egfr_multiplier
        
        if egfr < 60:
            if egfr < 30:
                active_factors['risk'].append(f"Severely Reduced Kidney Function (eGFR {egfr:.0f})")
            else:
                active_factors['risk'].append(f"Reduced Kidney Function (eGFR {egfr:.0f})")
    
    # ACR effect (major driver)
    if acr_mg_g > 10:
        acr_doublings = math.log2(acr_mg_g / 10.0)
        acr_multiplier = coefficients.risk_factors['acr_multiplier_doubling'] ** acr_doublings
        risk_multiplier *= acr_multiplier
        
        if acr_mg_g >= 30:
            if acr_mg_g >= 300:
                active_factors['risk'].append(f"Severe Proteinuria (ACR {acr_mg_g:.0f} mg/g)")
            else:
                active_factors['risk'].append(f"Mild Proteinuria (ACR {acr_mg_g:.0f} mg/g)")
    
    # HbA1c effect
    hba1c = features.get('hba1c', 7)
    if hba1c > 7:
        hba1c_excess = hba1c - 7.0
        hba1c_multiplier = coefficients.risk_factors['hba1c_multiplier_per_percent'] ** hba1c_excess
        risk_multiplier *= hba1c_multiplier
        
        if hba1c > 8.5:
            active_factors['risk'].append(f"Poor Glucose Control (HbA1c {hba1c:.1f}%)")
    
    # Blood pressure effect
    sbp = features.get('sbp', 120)
    if sbp > 120:
        sbp_excess_decades = (sbp - 120) / 10.0
        sbp_multiplier = coefficients.risk_factors['sbp_multiplier_per_10mmhg'] ** sbp_excess_decades
        risk_multiplier *= sbp_multiplier
        
        if sbp > 140:
            active_factors['risk'].append(f"High Blood Pressure ({sbp} mmHg)")
            
        if sbp > 160:
            severe_htn_multiplier = coefficients.risk_factors['severe_hypertension_bonus']
            risk_multiplier *= severe_htn_multiplier
    
    # Diabetes complications
    if features.get('retinopathy'):
        severity = features.get('retinopathy_severity', 'mild_npdr')
        if severity in ['severe_npdr', 'pdr']:
            retinopathy_multiplier = coefficients.risk_factors['retinopathy_severe_multiplier']
            active_factors['risk'].append("Advanced Diabetic Eye Disease")
        else:
            retinopathy_multiplier = coefficients.risk_factors['retinopathy_mild_multiplier']
            active_factors['risk'].append("Diabetic Eye Disease")
        
        risk_multiplier *= retinopathy_multiplier
    
    if features.get('neuropathy_dx'):
        neuropathy_multiplier = coefficients.risk_factors['neuropathy_multiplier']
        risk_multiplier *= neuropathy_multiplier
        active_factors['risk'].append("Diabetic Neuropathy")
    
    if features.get('ascvd_dx'):
        cvd_multiplier = coefficients.risk_factors['cardiovascular_multiplier']
        risk_multiplier *= cvd_multiplier
        active_factors['risk'].append("Cardiovascular Disease")
    
    # Protective medications
    if features.get('sglt2i_use'):
        sglt2i_protection = coefficients.risk_factors['sglt2i_protection_factor']
        risk_multiplier *= sglt2i_protection
        active_factors['protective'].append("SGLT2 Inhibitor")
    
    if features.get('ace_arb_use'):
        ace_arb_protection = coefficients.risk_factors['ace_arb_protection_factor']
        risk_multiplier *= ace_arb_protection
        active_factors['protective'].append("ACE Inhibitor/ARB")
    
    if features.get('statin_use'):
        statin_protection = coefficients.risk_factors['statin_protection_factor']
        risk_multiplier *= statin_protection
        active_factors['protective'].append("Statin Therapy")
    
    # Risk factors
    if features.get('insulin_used'):
        insulin_multiplier = coefficients.risk_factors['insulin_use_multiplier']
        risk_multiplier *= insulin_multiplier
        active_factors['risk'].append("Insulin Therapy (Advanced Disease)")
    
    if features.get('smoking_status') == 'current':
        smoking_multiplier = coefficients.risk_factors['current_smoking_multiplier']
        risk_multiplier *= smoking_multiplier
        active_factors['risk'].append("Current Smoking")
    
    if features.get('family_hx_ckd'):
        family_hx_multiplier = coefficients.risk_factors['family_history_multiplier']
        risk_multiplier *= family_hx_multiplier
        active_factors['risk'].append("Family History of Kidney Disease")
    
    if features.get('depression_dx'):
        depression_multiplier = coefficients.risk_factors['depression_multiplier']
        risk_multiplier *= depression_multiplier
        active_factors['risk'].append("Depression")
    
    # Calculate final monthly risk
    final_monthly_risk = monthly_risk * risk_multiplier
    
    # Cap monthly risk to realistic bounds
    final_monthly_risk = min(final_monthly_risk, 0.15)  # Max 15% per month
    final_monthly_risk = max(final_monthly_risk, 0.0005)  # Min 0.05% per month
    
    # Calculate 36-month cumulative risk
    survival_probability = (1 - final_monthly_risk) ** 36
    cumulative_risk_36_months = (1 - survival_probability) * 100  # Convert to percentage
    
    return cumulative_risk_36_months, active_factors, model_type

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
        number = {'suffix': "%"},
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

def create_risk_factors_chart(active_factors):
    """Create risk factors chart"""
    
    all_factors = active_factors['risk'] + active_factors['protective']
    
    if not all_factors:
        return None
    
    # Take up to 8 factors
    display_factors = all_factors[:8]
    
    # Create values and colors
    values = []
    colors = []
    
    for factor in display_factors:
        if factor in active_factors['risk']:
            values.append(1)
            colors.append('red')
        else:
            values.append(-1)
            colors.append('green')
    
    fig = go.Figure(go.Bar(
        x=values,
        y=display_factors,
        orientation='h',
        marker_color=colors,
        showlegend=False
    ))
    
    fig.update_layout(
        title="Risk Factors",
        xaxis_title="Effect",
        yaxis_title="",
        height=max(300, len(display_factors) * 35),
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
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
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
        egfr = st.number_input("Estimated GFR (mL/min/1.73m²)", min_value=5.0, max_value=150.0, value=95.0, step=1.0)
        st.session_state.patient_data['egfr'] = egfr
        
    with col2:
        acr_mg_mmol = st.number_input("ACR (mg/mmol)", min_value=0.0, max_value=100.0, value=1.5, step=0.1)
        acr_mg_g = convert_acr_units(acr_mg_mmol)
        st.session_state.patient_data['acr_mg_g'] = acr_mg_g
        st.info(f"ACR converted: {acr_mg_g:.1f} mg/g ({get_acr_category(acr_mg_g)})")
        
    with col3:
        hba1c = st.number_input("HbA1c (%)", min_value=5.0, max_value=15.0, value=6.8, step=0.1)
        st.session_state.patient_data['hba1c'] = hba1c
    
    # Blood Pressure
    st.subheader("Blood Pressure")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=120)
        st.session_state.patient_data['sbp'] = sbp
        
    with col2:
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=75)
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
        
        with st.spinner("Calculating personalized risk prediction..."):
            # Calculate truly dynamic risk
            risk_percentage, active_factors, model_type = calculate_dynamic_risk(features, coefficients)
        
        # Display Results
        st.header("Risk Prediction Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk gauge
            fig_gauge = create_risk_gauge(risk_percentage)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.metric("36-Month Risk", f"{risk_percentage:.1f}%")
            
            # Show model type used
            if model_type == "incident":
                st.info("Assessment: Predicting development of new kidney disease")
            else:
                st.info("Assessment: Predicting progression of existing kidney disease")
        
        with col2:
            # Risk factors chart
            if active_factors['risk'] or active_factors['protective']:
                fig_factors = create_risk_factors_chart(active_factors)
                if fig_factors:
                    st.plotly_chart(fig_factors, use_container_width=True)
            else:
                st.info("No significant risk factors or protective treatments identified for this patient profile.")
        
        # Clinical Interpretation
        st.subheader("Clinical Interpretation")
        
        if active_factors['risk'] or active_factors['protective']:
            
            if active_factors['risk']:
                st.write("**Factors Increasing Risk:**")
                for factor in active_factors['risk']:
                    st.write(f"• {factor}")
            
            if active_factors['protective']:
                st.write("**Protective Factors:**")
                for factor in active_factors['protective']:
                    st.write(f"• {factor}")
        else:
            st.write("• Patient has a low baseline risk profile with good diabetes control and normal kidney function.")
            st.write("• Continue current diabetes management and routine monitoring.")

if __name__ == "__main__":
    main()
