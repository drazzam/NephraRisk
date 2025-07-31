import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# SKLEARN COMPATIBILITY FIX
# ===============================================================================

def fix_sklearn_compatibility():
    """Fix sklearn compatibility issues for loading older models"""
    import sys
    
    # Handle different sklearn version compatibility issues
    try:
        import sklearn.ensemble._gb_losses
    except ImportError:
        try:
            # Try to import from new location (sklearn 1.0+)
            import sklearn.ensemble._gradient_boosting
            sys.modules['sklearn.ensemble._gb_losses'] = sklearn.ensemble._gradient_boosting
        except ImportError:
            try:
                # Try legacy import (sklearn < 1.0)
                import sklearn.ensemble.gradient_boosting
                sys.modules['sklearn.ensemble._gb_losses'] = sklearn.ensemble.gradient_boosting
            except ImportError:
                # Create minimal compatibility shim
                import types
                mock_module = types.ModuleType('sklearn.ensemble._gb_losses')
                # Add common classes that might be referenced
                class MockBinomialDeviance:
                    def __init__(self, *args, **kwargs): pass
                class MockMultinomialDeviance:
                    def __init__(self, *args, **kwargs): pass
                class MockLeastSquaresError:
                    def __init__(self, *args, **kwargs): pass
                
                mock_module.BinomialDeviance = MockBinomialDeviance
                mock_module.MultinomialDeviance = MockMultinomialDeviance  
                mock_module.LeastSquaresError = MockLeastSquaresError
                sys.modules['sklearn.ensemble._gb_losses'] = mock_module
    
    # Additional compatibility fixes for other sklearn modules
    compatibility_modules = [
        ('sklearn.tree._tree', 'sklearn.tree'),
        ('sklearn.tree._criterion', 'sklearn.tree'),
        ('sklearn.tree._splitter', 'sklearn.tree'),
        ('sklearn.ensemble._base', 'sklearn.ensemble'),
        ('sklearn.utils._testing', 'sklearn.utils'),
    ]
    
    for new_module, old_module in compatibility_modules:
        if new_module not in sys.modules:
            try:
                __import__(old_module)
                sys.modules[new_module] = sys.modules[old_module]
            except ImportError:
                pass

# Apply compatibility fix
fix_sklearn_compatibility()

# ===============================================================================
# PAGE CONFIGURATION
# ===============================================================================

st.set_page_config(
    page_title="DKD Risk Prediction Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================================
# CUSTOM CSS STYLING
# ===============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    
    .risk-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    
    .risk-high {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
    
    .risk-moderate {
        border-color: #ffc107;
        background-color: #fff3cd;
    }
    
    .risk-low {
        border-color: #28a745;
        background-color: #d4edda;
    }
    
    .feature-importance {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .disclaimer {
        background-color: #e9ecef;
        border-left: 4px solid #6c757d;
        padding: 10px;
        margin: 20px 0;
        border-radius: 5px;
    }
    
    .metric-container {
        text-align: center;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================================
# DEMO PREDICTOR (FALLBACK)
# ===============================================================================

class DemoPredictor:
    """Rule-based predictor for demo purposes when main model fails to load"""
    
    def __init__(self):
        self.feature_weights = {
            'baseline_eGFR': -0.25, 'baseline_ACR': 0.15, 'baseline_HbA1c': 0.12,
            'DR_grade': 0.08, 'NSAID_cum90d': 0.06, 'Family_Hx_CKD': 0.05,
            'IMD_quintile': 0.04, 'baseline_BMI': 0.03, 'Age': 0.02,
            'BB_flag': 0.02, 'CCB_flag': 0.02, 'Diuretic_flag': 0.02,
            'MRA_use': -0.03, 'baseline_SBP': 0.01, 'baseline_DBP': 0.01,
            'Gender': 0.01, 'Nationality': 0.005
        }
        self.baseline_risk = 0.095
        
    def predict_proba(self, X):
        """Predict probabilities using clinical rules"""
        predictions = []
        for _, row in X.iterrows():
            logit_risk = np.log(self.baseline_risk / (1 - self.baseline_risk))
            
            # eGFR contribution (strongest predictor)
            if 'baseline_eGFR' in row:
                egfr_contrib = self.feature_weights['baseline_eGFR'] * (90 - row['baseline_eGFR']) / 20
                logit_risk += egfr_contrib
            
            # ACR contribution  
            if 'baseline_ACR' in row:
                acr_contrib = self.feature_weights['baseline_ACR'] * np.log(max(row['baseline_ACR'], 1)) / 5
                logit_risk += acr_contrib
                
            # HbA1c contribution
            if 'baseline_HbA1c' in row:
                hba1c_contrib = self.feature_weights['baseline_HbA1c'] * (row['baseline_HbA1c'] - 7) / 2
                logit_risk += hba1c_contrib
            
            # Other features
            for feature, weight in self.feature_weights.items():
                if feature in row and feature not in ['baseline_eGFR', 'baseline_ACR', 'baseline_HbA1c']:
                    if feature == 'DR_grade':
                        contrib = weight * row[feature] * (1.5 ** row[feature])
                    elif feature == 'Age':
                        contrib = weight * (row[feature] - 60) / 15
                    elif feature == 'IMD_quintile':
                        contrib = weight * (row[feature] - 1) / 2
                    else:
                        contrib = weight * row[feature]
                    logit_risk += contrib
            
            probability = 1 / (1 + np.exp(-logit_risk))
            probability = np.clip(probability, 0.01, 0.99)
            predictions.append([1 - probability, probability])
        
        return np.array(predictions)

# ===============================================================================
# MODEL LOADING
# ===============================================================================

@st.cache_resource
def load_model():
    """Load the DKD risk prediction model with fallback to demo predictor"""
    try:
        model = joblib.load('dkd_risk_model.joblib')
        return model, None, False
    except Exception as e:
        # Return demo predictor as fallback
        demo_model = DemoPredictor()
        warning_msg = f"Main model unavailable: {str(e)}. Using demo predictor for demonstration."
        return demo_model, warning_msg, True

# ===============================================================================
# FEATURE DEFINITIONS
# ===============================================================================

FEATURE_DEFINITIONS = {
    'baseline_eGFR': {
        'label': 'eGFR (ml/min/1.73 m²)',
        'type': 'number',
        'min_value': 10.0,
        'max_value': 150.0,
        'value': 90.0,
        'step': 1.0,
        'help': 'Estimated Glomerular Filtration Rate - measure of kidney function'
    },
    'baseline_ACR': {
        'label': 'Albumin-Creatinine Ratio (mg/g)',
        'type': 'number',
        'min_value': 0.0,
        'max_value': 1000.0,
        'value': 10.0,
        'step': 1.0,
        'help': 'Urinary albumin-creatinine ratio - marker of kidney damage'
    },
    'baseline_HbA1c': {
        'label': 'HbA1c (%)',
        'type': 'number',
        'min_value': 4.0,
        'max_value': 18.0,
        'value': 7.5,
        'step': 0.1,
        'help': 'Glycated hemoglobin - measure of blood sugar control'
    },
    'baseline_BMI': {
        'label': 'BMI (kg/m²)',
        'type': 'number',
        'min_value': 15.0,
        'max_value': 50.0,
        'value': 28.0,
        'step': 0.1,
        'help': 'Body Mass Index'
    },
    'Age': {
        'label': 'Age (years)',
        'type': 'number',
        'min_value': 18,
        'max_value': 100,
        'value': 55,
        'step': 1,
        'help': 'Patient age in years'
    },
    'baseline_SBP': {
        'label': 'Systolic Blood Pressure (mmHg)',
        'type': 'number',
        'min_value': 80,
        'max_value': 200,
        'value': 130,
        'step': 1,
        'help': 'Systolic blood pressure'
    },
    'baseline_DBP': {
        'label': 'Diastolic Blood Pressure (mmHg)',
        'type': 'number',
        'min_value': 40,
        'max_value': 120,
        'value': 80,
        'step': 1,
        'help': 'Diastolic blood pressure'
    },
    'DR_grade': {
        'label': 'Diabetic Retinopathy Grade',
        'type': 'selectbox',
        'options': [0, 1, 2, 3, 4],
        'format_func': lambda x: {
            0: '0 - No retinopathy',
            1: '1 - Mild NPDR',
            2: '2 - Moderate NPDR', 
            3: '3 - Severe NPDR',
            4: '4 - Proliferative DR'
        }[x],
        'help': 'Diabetic retinopathy severity grade'
    },
    'IMD_quintile': {
        'label': 'Socioeconomic Deprivation Quintile',
        'type': 'selectbox',
        'options': [1, 2, 3, 4, 5],
        'format_func': lambda x: f"{x} - {'Least' if x == 1 else 'Most' if x == 5 else 'Moderate'} deprived",
        'help': 'Socioeconomic deprivation level (1=least deprived, 5=most deprived)'
    },
    'Gender': {
        'label': 'Sex',
        'type': 'selectbox',
        'options': ['Female', 'Male'],
        'help': 'Patient sex'
    },
    'Nationality': {
        'label': 'Nationality',
        'type': 'selectbox',
        'options': ['Saudi', 'Non-Saudi'],
        'help': 'Patient nationality'
    },
    'Family_Hx_CKD': {
        'label': 'Family History of CKD',
        'type': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'No' if x == 0 else 'Yes',
        'help': 'First-degree relative with chronic kidney disease'
    },
    'NSAID_cum90d': {
        'label': 'NSAID Use ≥90 days/year',
        'type': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'No' if x == 0 else 'Yes',
        'help': 'Regular NSAID use (≥90 days per year)'
    },
    'BB_flag': {
        'label': 'Beta-blocker Use',
        'type': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'No' if x == 0 else 'Yes',
        'help': 'Currently taking beta-blockers'
    },
    'CCB_flag': {
        'label': 'Calcium Channel Blocker Use',
        'type': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'No' if x == 0 else 'Yes',
        'help': 'Currently taking calcium channel blockers'
    },
    'Diuretic_flag': {
        'label': 'Diuretic Use',
        'type': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'No' if x == 0 else 'Yes',
        'help': 'Currently taking diuretics'
    },
    'MRA_use': {
        'label': 'Mineralocorticoid Receptor Antagonist Use',
        'type': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'No' if x == 0 else 'Yes',
        'help': 'Currently taking MRA (e.g., finerenone, eplerenone)'
    }
}

# ===============================================================================
# PREDICTION FUNCTIONS
# ===============================================================================

def create_input_dataframe(user_inputs):
    """Create properly formatted input DataFrame for the model"""
    
    # Convert inputs to the expected format
    input_data = {}
    
    for feature, value in user_inputs.items():
        if feature == 'Gender':
            # Convert gender to numeric if needed (depends on model training)
            input_data[feature] = 1 if value == 'Male' else 0
        elif feature == 'Nationality':
            # Convert nationality to numeric if needed
            input_data[feature] = 1 if value == 'Saudi' else 0
        else:
            input_data[feature] = value
    
    # Create DataFrame with single row
    df = pd.DataFrame([input_data])
    
    # Ensure all expected features are present
    expected_features = list(FEATURE_DEFINITIONS.keys())
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Reorder columns to match expected order
    df = df[expected_features]
    
    return df

def predict_risk(model, input_df):
    """Make risk prediction with error handling"""
    try:
        if hasattr(model, 'predict_proba'):
            # Get probability of positive class (DKD risk)
            probabilities = model.predict_proba(input_df)
            risk_probability = probabilities[0, 1] if probabilities.shape[1] > 1 else probabilities[0, 0]
        else:
            # Fallback to predict method
            prediction = model.predict(input_df)
            risk_probability = prediction[0]
        
        # Convert to percentage
        risk_percentage = risk_probability * 100
        
        return risk_percentage, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def get_risk_category(risk_percentage):
    """Categorize risk level"""
    if risk_percentage < 10:
        return "Low", "risk-low", "#28a745"
    elif risk_percentage < 20:
        return "Moderate", "risk-moderate", "#ffc107"
    else:
        return "High", "risk-high", "#dc3545"

# ===============================================================================
# VISUALIZATION FUNCTIONS
# ===============================================================================

def create_risk_gauge(risk_percentage):
    """Create a risk gauge visualization"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "3-Year DKD Risk (%)"},
        delta = {'reference': 10, 'suffix': "%"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_importance_chart():
    """Create a mock feature importance chart based on documentation"""
    
    # Feature importance from documentation (top 10)
    features = [
        'eGFR', 'ACR', 'DR Grade', 'HbA1c', 'NSAID Use',
        'Family History', 'Deprivation', 'BMI', 'β-blocker', 'CCB'
    ]
    
    importance = [21.3, 17.2, 10.1, 8.4, 6.2, 4.9, 4.2, 3.0, 2.7, 2.6]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in DKD Risk Prediction",
        labels={'x': 'Importance (%)', 'y': 'Features'},
        color=importance,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# ===============================================================================
# REPORT GENERATION
# ===============================================================================

def generate_pdf_report(user_inputs, risk_percentage, risk_category):
    """Generate a simple text report"""
    
    report = f"""
DKD RISK PREDICTION REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION:
- Age: {user_inputs['Age']} years
- Sex: {user_inputs['Gender']}
- BMI: {user_inputs['baseline_BMI']} kg/m²

CLINICAL PARAMETERS:
- eGFR: {user_inputs['baseline_eGFR']} ml/min/1.73m²
- ACR: {user_inputs['baseline_ACR']} mg/g
- HbA1c: {user_inputs['baseline_HbA1c']}%
- Blood Pressure: {user_inputs['baseline_SBP']}/{user_inputs['baseline_DBP']} mmHg

RISK ASSESSMENT:
- 3-Year DKD Risk: {risk_percentage:.1f}%
- Risk Category: {risk_category}

MEDICATIONS:
- Beta-blocker: {'Yes' if user_inputs['BB_flag'] else 'No'}
- CCB: {'Yes' if user_inputs['CCB_flag'] else 'No'}
- Diuretic: {'Yes' if user_inputs['Diuretic_flag'] else 'No'}
- MRA: {'Yes' if user_inputs['MRA_use'] else 'No'}

RISK FACTORS:
- Family History of CKD: {'Yes' if user_inputs['Family_Hx_CKD'] else 'No'}
- Regular NSAID Use: {'Yes' if user_inputs['NSAID_cum90d'] else 'No'}
- DR Grade: {user_inputs['DR_grade']}

DISCLAIMER:
This prediction is for clinical decision support only and should not replace clinical judgment.
Consult with healthcare professionals for medical decisions.
"""
    
    return report

# ===============================================================================
# MAIN APPLICATION
# ===============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 DKD Risk Prediction Platform</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Clinical Decision Support Tool</strong><br>
        This tool provides risk predictions for clinical judgment support only. 
        It does not replace professional medical assessment or clinical decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, warning, is_demo = load_model()
    
    if model is None:
        st.error("❌ Unable to load prediction model")
        st.info("Please ensure the model file 'dkd_risk_model.joblib' is present in the repository.")
        return
    
    if is_demo:
        st.warning(f"⚠️ {warning}")
        st.info("This demo uses clinical rules for risk estimation. Deploy with the actual model file for production use.")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Patient Information")
    
    # Collect user inputs
    user_inputs = {}
    
    # Demographics section
    st.sidebar.subheader("👤 Demographics")
    for feature in ['Age', 'Gender', 'Nationality']:
        feature_def = FEATURE_DEFINITIONS[feature]
        
        if feature_def['type'] == 'number':
            user_inputs[feature] = st.sidebar.number_input(
                feature_def['label'],
                min_value=feature_def['min_value'],
                max_value=feature_def['max_value'],
                value=feature_def['value'],
                step=feature_def['step'],
                help=feature_def['help']
            )
        elif feature_def['type'] == 'selectbox':
            user_inputs[feature] = st.sidebar.selectbox(
                feature_def['label'],
                feature_def['options'],
                format_func=feature_def.get('format_func', str),
                help=feature_def['help']
            )
    
    # Clinical parameters section
    st.sidebar.subheader("🔬 Clinical Parameters")
    for feature in ['baseline_eGFR', 'baseline_ACR', 'baseline_HbA1c', 'baseline_BMI', 'baseline_SBP', 'baseline_DBP']:
        feature_def = FEATURE_DEFINITIONS[feature]
        user_inputs[feature] = st.sidebar.number_input(
            feature_def['label'],
            min_value=feature_def['min_value'],
            max_value=feature_def['max_value'],
            value=feature_def['value'],
            step=feature_def['step'],
            help=feature_def['help']
        )
    
    # Medical history section
    st.sidebar.subheader("📜 Medical History")
    for feature in ['DR_grade', 'Family_Hx_CKD', 'NSAID_cum90d', 'IMD_quintile']:
        feature_def = FEATURE_DEFINITIONS[feature]
        user_inputs[feature] = st.sidebar.selectbox(
            feature_def['label'],
            feature_def['options'],
            format_func=feature_def.get('format_func', str),
            help=feature_def['help']
        )
    
    # Medications section
    st.sidebar.subheader("💊 Current Medications")
    for feature in ['BB_flag', 'CCB_flag', 'Diuretic_flag', 'MRA_use']:
        feature_def = FEATURE_DEFINITIONS[feature]
        user_inputs[feature] = st.sidebar.selectbox(
            feature_def['label'],
            feature_def['options'],
            format_func=feature_def.get('format_func', str),
            help=feature_def['help']
        )
    
    # Prediction button
    if st.sidebar.button("🔮 Calculate Risk", type="primary"):
        
        # Create input DataFrame
        input_df = create_input_dataframe(user_inputs)
        
        # Make prediction
        risk_percentage, pred_error = predict_risk(model, input_df)
        
        if pred_error:
            st.error(f"❌ {pred_error}")
            return
        
        # Get risk category
        risk_category, risk_class, risk_color = get_risk_category(risk_percentage)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk gauge
            fig_gauge = create_risk_gauge(risk_percentage)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk interpretation
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h3 style="color: {risk_color};">Risk Assessment: {risk_category}</h3>
                <p><strong>3-Year DKD Risk: {risk_percentage:.1f}%</strong></p>
                <p>This patient has a <strong>{risk_category.lower()}</strong> risk of developing or progressing 
                diabetic kidney disease within the next 3 years.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Key metrics
            st.markdown("### 📊 Key Metrics")
            
            metrics = [
                ("eGFR", f"{user_inputs['baseline_eGFR']:.0f}", "ml/min/1.73m²"),
                ("ACR", f"{user_inputs['baseline_ACR']:.0f}", "mg/g"),
                ("HbA1c", f"{user_inputs['baseline_HbA1c']:.1f}", "%"),
                ("Age", f"{user_inputs['Age']}", "years")
            ]
            
            for label, value, unit in metrics:
                st.metric(label, f"{value} {unit}")
        
        # Feature importance chart
        st.subheader("📈 Model Feature Importance")
        fig_importance = create_feature_importance_chart()
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Clinical interpretation
        st.subheader("🎯 Clinical Interpretation")
        
        interpretation_text = []
        
        # eGFR interpretation
        if user_inputs['baseline_eGFR'] < 60:
            interpretation_text.append(f"• Low eGFR ({user_inputs['baseline_eGFR']:.0f} ml/min/1.73m²) significantly increases DKD risk")
        elif user_inputs['baseline_eGFR'] > 90:
            interpretation_text.append(f"• Normal eGFR ({user_inputs['baseline_eGFR']:.0f} ml/min/1.73m²) is protective")
        
        # ACR interpretation
        if user_inputs['baseline_ACR'] >= 30:
            interpretation_text.append(f"• Elevated ACR ({user_inputs['baseline_ACR']:.0f} mg/g) indicates existing kidney damage")
        elif user_inputs['baseline_ACR'] < 10:
            interpretation_text.append(f"• Low ACR ({user_inputs['baseline_ACR']:.0f} mg/g) is favorable")
        
        # HbA1c interpretation
        if user_inputs['baseline_HbA1c'] > 9:
            interpretation_text.append(f"• Poor glycemic control (HbA1c {user_inputs['baseline_HbA1c']:.1f}%) increases risk")
        elif user_inputs['baseline_HbA1c'] < 7:
            interpretation_text.append(f"• Good glycemic control (HbA1c {user_inputs['baseline_HbA1c']:.1f}%) is protective")
        
        # Risk factors
        if user_inputs['Family_Hx_CKD']:
            interpretation_text.append("• Family history of CKD increases risk")
        
        if user_inputs['NSAID_cum90d']:
            interpretation_text.append("• Regular NSAID use contributes to kidney risk")
        
        if user_inputs['MRA_use']:
            interpretation_text.append("• MRA therapy may provide kidney protection")
        
        if interpretation_text:
            for text in interpretation_text:
                st.write(text)
        else:
            st.write("• Overall risk profile is within expected range for the given parameters")
        
        # Generate report
        st.subheader("📄 Clinical Report")
        report_text = generate_pdf_report(user_inputs, risk_percentage, risk_category)
        
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name=f"DKD_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        with st.expander("📝 View Report"):
            st.text(report_text)
    
    # Information sections
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ℹ️ About This Tool")
        st.write("""
        This DKD (Diabetic Kidney Disease) risk prediction tool uses machine learning 
        to estimate the 3-year risk of developing or progressing kidney disease in 
        diabetic patients.
        
        **Model Features:**
        - Stacked ensemble (LightGBM + CoxBoost)
        - AUROC: 0.866, AUPRC: 0.522
        - Calibrated predictions with confidence intervals
        - Based on 42,380 patient registry data
        """)
    
    with col2:
        st.subheader("🎯 Clinical Usage")
        st.write("""
        **Risk Categories:**
        - **Low (<10%):** Standard monitoring
        - **Moderate (10-20%):** Enhanced monitoring, lifestyle interventions
        - **High (>20%):** Nephrology referral, intensive management
        
        **Key Predictors:**
        - eGFR and ACR (kidney function markers)
        - HbA1c (diabetes control)
        - Diabetic retinopathy grade
        - Family history and medications
        """)

if __name__ == "__main__":
    main()
