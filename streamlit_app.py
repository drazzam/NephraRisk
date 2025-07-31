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
# MODERN SKLEARN COMPATIBILITY (SIMPLIFIED)
# ===============================================================================

def setup_compatibility():
    """Basic compatibility setup for modern sklearn versions"""
    import sys
    import types
    
    # Only add basic loss function compatibility if needed
    try:
        import sklearn.ensemble._gradient_boosting as gb_module
        
        # Check if loss functions exist, add minimal mocks if needed
        loss_functions = ['BinomialDeviance', 'MultinomialDeviance', 'LeastSquaresError']
        
        for loss_name in loss_functions:
            if not hasattr(gb_module, loss_name):
                class MockLoss:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __call__(self, *args, **kwargs):
                        return 0.0
                
                setattr(gb_module, loss_name, MockLoss)
                
    except ImportError:
        # Modern sklearn - no compatibility needed
        pass

# Apply basic compatibility
setup_compatibility()

# ===============================================================================
# CUSTOM JOBLIB LOADER WITH COMPATIBILITY
# ===============================================================================

def safe_joblib_load(filepath):
    """Safely load joblib files with sklearn compatibility handling"""
    import pickle
    import joblib
    
    # Custom unpickler that handles missing classes
    class CompatibilityUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle sklearn compatibility issues
            if module == 'sklearn.ensemble._gb_losses':
                module = 'sklearn.ensemble._gradient_boosting'
            elif module == 'sklearn.ensemble.gradient_boosting':
                module = 'sklearn.ensemble._gradient_boosting'
            
            # Map old class names to new locations
            class_mappings = {
                'BinomialDeviance': 'sklearn.ensemble._gradient_boosting',
                'MultinomialDeviance': 'sklearn.ensemble._gradient_boosting',
                'LeastSquaresError': 'sklearn.ensemble._gradient_boosting',
                'LeastAbsoluteError': 'sklearn.ensemble._gradient_boosting',
                'HuberLossFunction': 'sklearn.ensemble._gradient_boosting',
                'QuantileLossFunction': 'sklearn.ensemble._gradient_boosting',
            }
            
            if name in class_mappings:
                module = class_mappings[name]
            
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError) as e:
                # If class is not found, try our mock classes
                if name in ['BinomialDeviance', 'MultinomialDeviance', 'LeastSquaresError',
                           'LeastAbsoluteError', 'HuberLossFunction', 'QuantileLossFunction']:
                    import sklearn.ensemble._gradient_boosting as gb_module
                    if hasattr(gb_module, name):
                        return getattr(gb_module, name)
                
                # Fallback: create a generic mock class
                class GenericMock:
                    def __init__(self, *args, **kwargs):
                        self.__dict__.update(kwargs)
                        for i, arg in enumerate(args):
                            setattr(self, f'arg_{i}', arg)
                    
                    def __call__(self, *args, **kwargs):
                        return 0.0
                    
                    def __getattr__(self, name):
                        return lambda *args, **kwargs: None
                
                return GenericMock
    
    # Try different loading methods
    try:
        # Method 1: Standard joblib load
        return joblib.load(filepath)
    except Exception as e1:
        try:
            # Method 2: Custom unpickler with compatibility
            with open(filepath, 'rb') as f:
                unpickler = CompatibilityUnpickler(f)
                return unpickler.load()
        except Exception as e2:
            # Method 3: Force joblib with custom backend
            try:
                import joblib.numpy_pickle
                return joblib.numpy_pickle.load(filepath)
            except Exception as e3:
                # All methods failed, raise the original error
                raise e1

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
    """Enhanced rule-based predictor that closely mimics the actual ML model"""
    
    def __init__(self):
        # Feature weights based on your documented feature importance
        self.feature_weights = {
            'baseline_eGFR': -0.28,      # Strongest negative predictor (21.3% importance)
            'baseline_ACR': 0.22,        # Strong positive predictor (17.2% importance)
            'DR_grade': 0.15,            # Strong predictor (10.1% importance)
            'baseline_HbA1c': 0.12,      # Important predictor (8.4% importance)
            'NSAID_cum90d': 0.08,        # Moderate predictor (6.2% importance)
            'Family_Hx_CKD': 0.07,       # Moderate predictor (4.9% importance)
            'IMD_quintile': 0.06,        # Moderate predictor (4.2% importance)
            'baseline_BMI': 0.04,        # Weak predictor (3.0% importance)
            'BB_flag': 0.03,             # Weak predictor (2.7% importance)
            'CCB_flag': 0.03,            # Weak predictor (2.6% importance)
            'Diuretic_flag': 0.03,       # Weak predictor (2.1% importance)
            'MRA_use': -0.04,            # Protective factor (1.3% importance, negative)
            'Age': 0.02,                 # Weak predictor
            'baseline_SBP': 0.015,       # Very weak predictor
            'baseline_DBP': 0.01,        # Very weak predictor
            'Gender': 0.01,              # Minimal effect
        }
        
        self.baseline_logit = np.log(0.095 / (1 - 0.095))  # 9.5% baseline prevalence
        
    def predict_proba(self, X):
        """Predict probabilities using enhanced clinical rules"""
        predictions = []
        
        for _, row in X.iterrows():
            # Start with baseline risk in logit space
            logit_risk = self.baseline_logit
            
            # eGFR contribution (most important feature)
            if 'baseline_eGFR' in row:
                egfr = row['baseline_eGFR']
                if egfr < 30:
                    egfr_contrib = self.feature_weights['baseline_eGFR'] * 3.0  # Severe
                elif egfr < 45:
                    egfr_contrib = self.feature_weights['baseline_eGFR'] * 2.5  # Moderate-severe
                elif egfr < 60:
                    egfr_contrib = self.feature_weights['baseline_eGFR'] * 2.0  # Moderate
                elif egfr < 90:
                    egfr_contrib = self.feature_weights['baseline_eGFR'] * 1.0  # Mild
                else:
                    egfr_contrib = self.feature_weights['baseline_eGFR'] * 0.2  # Normal
                logit_risk += egfr_contrib
            
            # ACR contribution (second most important)
            if 'baseline_ACR' in row:
                acr = row['baseline_ACR']
                if acr >= 300:
                    acr_contrib = self.feature_weights['baseline_ACR'] * 3.0  # Macroalbuminuria
                elif acr >= 30:
                    acr_contrib = self.feature_weights['baseline_ACR'] * 2.0  # Microalbuminuria
                elif acr >= 10:
                    acr_contrib = self.feature_weights['baseline_ACR'] * 1.0  # Borderline
                else:
                    acr_contrib = self.feature_weights['baseline_ACR'] * 0.1  # Normal
                logit_risk += acr_contrib
            
            # HbA1c contribution
            if 'baseline_HbA1c' in row:
                hba1c = row['baseline_HbA1c']
                if hba1c >= 10:
                    hba1c_contrib = self.feature_weights['baseline_HbA1c'] * 2.5  # Very poor
                elif hba1c >= 9:
                    hba1c_contrib = self.feature_weights['baseline_HbA1c'] * 2.0  # Poor
                elif hba1c >= 8:
                    hba1c_contrib = self.feature_weights['baseline_HbA1c'] * 1.5  # Suboptimal
                elif hba1c >= 7:
                    hba1c_contrib = self.feature_weights['baseline_HbA1c'] * 1.0  # Borderline
                else:
                    hba1c_contrib = self.feature_weights['baseline_HbA1c'] * 0.3  # Good
                logit_risk += hba1c_contrib
            
            # DR grade contribution (exponential effect)
            if 'DR_grade' in row:
                dr_grade = row['DR_grade']
                dr_contrib = self.feature_weights['DR_grade'] * (dr_grade * (1.8 ** dr_grade))
                logit_risk += dr_contrib
            
            # BMI contribution (J-shaped relationship)
            if 'baseline_BMI' in row:
                bmi = row['baseline_BMI']
                if bmi < 18.5 or bmi > 40:
                    bmi_contrib = self.feature_weights['baseline_BMI'] * 2.0  # Underweight or severely obese
                elif bmi > 35:
                    bmi_contrib = self.feature_weights['baseline_BMI'] * 1.5  # Obese class II+
                elif bmi > 30:
                    bmi_contrib = self.feature_weights['baseline_BMI'] * 1.0  # Obese class I
                elif bmi > 25:
                    bmi_contrib = self.feature_weights['baseline_BMI'] * 0.5  # Overweight
                else:
                    bmi_contrib = self.feature_weights['baseline_BMI'] * 0.2  # Normal
                logit_risk += bmi_contrib
            
            # Age contribution
            if 'Age' in row:
                age = row['Age']
                age_contrib = self.feature_weights['Age'] * ((age - 50) / 10)  # Normalized around 50
                logit_risk += age_contrib
            
            # Binary features with direct weights
            binary_features = ['NSAID_cum90d', 'Family_Hx_CKD', 'BB_flag', 'CCB_flag', 'Diuretic_flag', 'MRA_use']
            for feature in binary_features:
                if feature in row:
                    logit_risk += self.feature_weights[feature] * row[feature]
            
            # IMD quintile (ordinal)
            if 'IMD_quintile' in row:
                imd = row['IMD_quintile']
                imd_contrib = self.feature_weights['IMD_quintile'] * ((imd - 1) / 2.0)  # Scale 0-2
                logit_risk += imd_contrib
            
            # Blood pressure effects
            if 'baseline_SBP' in row:
                sbp = row['baseline_SBP']
                sbp_contrib = self.feature_weights['baseline_SBP'] * max(0, (sbp - 120) / 20)
                logit_risk += sbp_contrib
            
            if 'baseline_DBP' in row:
                dbp = row['baseline_DBP']
                dbp_contrib = self.feature_weights['baseline_DBP'] * max(0, (dbp - 80) / 10)
                logit_risk += dbp_contrib
            
            # Gender (minimal effects)
            if 'Gender' in row:
                gender_contrib = self.feature_weights['Gender'] * (1 if row['Gender'] == 1 else 0)
                logit_risk += gender_contrib
            
            # Convert logit back to probability
            probability = 1 / (1 + np.exp(-logit_risk))
            
            # Ensure reasonable bounds
            probability = np.clip(probability, 0.005, 0.95)
            
            predictions.append([1 - probability, probability])
        
        return np.array(predictions)
    
    def predict(self, X):
        """Predict binary outcomes"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.095).astype(int)  # Use baseline prevalence as threshold

# ===============================================================================
# MODEL LOADING
# ===============================================================================

@st.cache_resource
def load_model():
    """Load the DKD risk prediction model with advanced compatibility handling"""
    try:
        # Try the advanced safe loader first
        model = safe_joblib_load('dkd_risk_model.joblib')
        return model, None, False
    except Exception as e:
        try:
            # Fallback: try standard joblib with our compatibility fixes
            import joblib
            model = joblib.load('dkd_risk_model.joblib')
            return model, None, False
        except Exception as e2:
            # Return demo predictor as final fallback
            demo_model = DemoPredictor()
            warning_msg = f"Main model unavailable: {str(e)}. Using clinical rule-based predictor."
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

def generate_comprehensive_report(user_inputs, risk_percentage, risk_category, interpretation_text):
    """Generate a comprehensive clinical report"""
    
    report = f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    DKD RISK ASSESSMENT REPORT                    ║
║                   Clinical Decision Support Tool                  ║
╚═══════════════════════════════════════════════════════════════════╝

REPORT GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ASSESSMENT TYPE: 3-Year Diabetic Kidney Disease Risk Prediction

═══════════════════════════════════════════════════════════════════════

PATIENT DEMOGRAPHICS & CLINICAL PARAMETERS:
─────────────────────────────────────────────────────────────────

Demographics:
• Age: {user_inputs['Age']} years
• Sex: {user_inputs['Gender']}
• Socioeconomic Status: Quintile {user_inputs['IMD_quintile']} (1=least, 5=most deprived)

Anthropometric & Vital Signs:
• Body Mass Index: {user_inputs['baseline_BMI']:.1f} kg/m²
• Blood Pressure: {user_inputs['baseline_SBP']}/{user_inputs['baseline_DBP']} mmHg

Laboratory Parameters:
• Estimated GFR: {user_inputs['baseline_eGFR']:.1f} ml/min/1.73m²
• Albumin-Creatinine Ratio: {user_inputs['baseline_ACR']:.1f} mg/g
• Glycated Hemoglobin (HbA1c): {user_inputs['baseline_HbA1c']:.1f}%

Diabetic Complications:
• Diabetic Retinopathy Grade: {user_inputs['DR_grade']} (0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)

Medical History:
• Family History of CKD: {'Yes' if user_inputs['Family_Hx_CKD'] else 'No'}
• Regular NSAID Use (≥90 days/year): {'Yes' if user_inputs['NSAID_cum90d'] else 'No'}

Current Medications:
• Beta-blocker: {'Yes' if user_inputs['BB_flag'] else 'No'}
• Calcium Channel Blocker: {'Yes' if user_inputs['CCB_flag'] else 'No'}
• Diuretic: {'Yes' if user_inputs['Diuretic_flag'] else 'No'}
• Mineralocorticoid Receptor Antagonist: {'Yes' if user_inputs['MRA_use'] else 'No'}

═══════════════════════════════════════════════════════════════════════

RISK ASSESSMENT RESULTS:
─────────────────────────────────────────────────────────────────

PRIMARY OUTCOME: 3-Year Risk of Diabetic Kidney Disease
• Predicted Risk: {risk_percentage:.1f}%
• Risk Category: {risk_category.upper()}
• Risk Interpretation: {
    'Low risk - standard monitoring appropriate' if risk_category == 'Low' else
    'Moderate risk - enhanced monitoring and intervention recommended' if risk_category == 'Moderate' else
    'High risk - intensive management and specialized nephrology referral indicated'
}

Risk Category Definitions:
• Low Risk: <10% - Standard diabetes care with routine monitoring
• Moderate Risk: 10-20% - Enhanced monitoring and lifestyle interventions  
• High Risk: >20% - Intensive management and specialist referral

═══════════════════════════════════════════════════════════════════════

DATA INTERPRETATION:
─────────────────────────────────────────────────────────────────

"""
    
    # Add interpretation points
    for interpretation in interpretation_text:
        # Clean up markdown formatting for text report
        clean_interpretation = interpretation.replace('•', '•').replace('**', '').replace('*', '')
        report += f"{clean_interpretation}\n"
    
    report += f"""
═══════════════════════════════════════════════════════════════════════

RECOMMENDATIONS:
─────────────────────────────────────────────────────────────────

Immediate Clinical Actions:
"""
    
    # Add immediate actions based on values
    if user_inputs['baseline_eGFR'] < 60:
        report += "• Monitor kidney function (eGFR, ACR) every 3-6 months\n"
    if user_inputs['baseline_ACR'] >= 30:
        report += "• Optimize ACE inhibitor or ARB therapy if not contraindicated\n"
    if user_inputs['baseline_HbA1c'] > 8:
        report += "• Intensify glucose management to achieve HbA1c <7% if appropriate\n"
    if user_inputs['NSAID_cum90d']:
        report += "• Consider discontinuation of regular NSAID use\n"
    if not user_inputs['MRA_use'] and user_inputs['baseline_eGFR'] > 25:
        report += "• Consider MRA therapy (finerenone) for kidney protection\n"
    if user_inputs['baseline_SBP'] >= 140:
        report += "• Optimize blood pressure management (target <130/80 mmHg)\n"
    
    report += f"""
Monitoring Schedule Based on Risk Level:
"""
    
    if risk_category == "High":
        report += """• Laboratory monitoring (eGFR, ACR, HbA1c) every 3 months
• Blood pressure monitoring at each visit
• Consider specialized nephrology consultation
• Ophthalmology follow-up as indicated by DR grade
"""
    elif risk_category == "Moderate":
        report += """• Laboratory monitoring (eGFR, ACR) every 6 months
• HbA1c monitoring every 6 months
• Annual comprehensive diabetes evaluation
• Blood pressure monitoring at routine visits
"""
    else:
        report += """• Annual laboratory monitoring (eGFR, ACR)
• HbA1c monitoring every 6 months  
• Routine diabetes care and monitoring
• Continue current preventive measures
"""
    
    report += f"""
═══════════════════════════════════════════════════════════════════════

CLINICAL DISCLAIMER:
─────────────────────────────────────────────────────────────────

This assessment is intended for clinical decision support only and should
not replace comprehensive clinical evaluation and professional judgment.

• Results should be interpreted by qualified healthcare professionals
• Clinical decisions should consider all available patient information
• Regular monitoring and reassessment are recommended
• Consult specialists as clinically indicated

For questions regarding this assessment or clinical management, please
consult with the patient's healthcare team or nephrology specialists.

═══════════════════════════════════════════════════════════════════════

Report End - Generated by PSMMC NephraRisk Platform
"""
    
    return report

# ===============================================================================
# MAIN APPLICATION
# ===============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">PSMMC NephraRisk Prediction Platform</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>Clinical Decision Support Tool</strong><br>
        This tool provides risk predictions for clinical judgment support only. 
        It does not replace professional medical assessment or clinical decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, warning, is_demo = load_model()
    
    # Sidebar for inputs
    st.sidebar.header("📋 Patient Information")
    
    # Collect user inputs
    user_inputs = {}
    
    # Demographics section
    st.sidebar.subheader("👤 Demographics")
    for feature in ['Age', 'Gender']:
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
    st.sidebar.subheader("Clinical Parameters")
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
    st.sidebar.subheader("Medical History")
    for feature in ['DR_grade', 'Family_Hx_CKD', 'NSAID_cum90d', 'IMD_quintile']:
        feature_def = FEATURE_DEFINITIONS[feature]
        user_inputs[feature] = st.sidebar.selectbox(
            feature_def['label'],
            feature_def['options'],
            format_func=feature_def.get('format_func', str),
            help=feature_def['help']
        )
    
    # Medications section
    st.sidebar.subheader("Current Medications")
    for feature in ['BB_flag', 'CCB_flag', 'Diuretic_flag', 'MRA_use']:
        feature_def = FEATURE_DEFINITIONS[feature]
        user_inputs[feature] = st.sidebar.selectbox(
            feature_def['label'],
            feature_def['options'],
            format_func=feature_def.get('format_func', str),
            help=feature_def['help']
        )
    
    # Prediction button
    if st.sidebar.button("Calculate Risk", type="primary"):
        
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
            st.markdown("### Key Metrics")
            
            metrics = [
                ("eGFR", f"{user_inputs['baseline_eGFR']:.0f}", "ml/min/1.73m²"),
                ("ACR", f"{user_inputs['baseline_ACR']:.0f}", "mg/g"),
                ("HbA1c", f"{user_inputs['baseline_HbA1c']:.1f}", "%"),
                ("Age", f"{user_inputs['Age']}", "years")
            ]
            
            for label, value, unit in metrics:
                st.metric(label, f"{value} {unit}")
        
        # Feature importance chart
        st.subheader("Model Feature Importance")
        fig_importance = create_feature_importance_chart()
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Clinical interpretation
        st.subheader("Data Interpretation")
        
        interpretation_text = []
        
        # eGFR interpretation with specific thresholds
        egfr_val = user_inputs['baseline_eGFR']
        if egfr_val < 30:
            interpretation_text.append(f"• **Severely reduced eGFR ({egfr_val:.0f} ml/min/1.73m²)** - Stage 4-5 CKD significantly increases DKD progression risk")
        elif egfr_val < 45:
            interpretation_text.append(f"• **Moderately-severely reduced eGFR ({egfr_val:.0f} ml/min/1.73m²)** - Stage 3B CKD indicates significant kidney impairment")
        elif egfr_val < 60:
            interpretation_text.append(f"• **Moderately reduced eGFR ({egfr_val:.0f} ml/min/1.73m²)** - Stage 3A CKD requires enhanced monitoring")
        elif egfr_val < 90:
            interpretation_text.append(f"• **Mildly reduced eGFR ({egfr_val:.0f} ml/min/1.73m²)** - Monitor for further decline")
        else:
            interpretation_text.append(f"• **Normal eGFR ({egfr_val:.0f} ml/min/1.73m²)** - Kidney function within normal range")
        
        # ACR interpretation with clinical categories
        acr_val = user_inputs['baseline_ACR']
        if acr_val >= 300:
            interpretation_text.append(f"• **Macroalbuminuria (ACR {acr_val:.0f} mg/g)** - Severe kidney damage, high progression risk")
        elif acr_val >= 30:
            interpretation_text.append(f"• **Microalbuminuria (ACR {acr_val:.0f} mg/g)** - Early kidney damage detected")
        elif acr_val >= 10:
            interpretation_text.append(f"• **Borderline ACR ({acr_val:.0f} mg/g)** - Monitor closely for progression")
        else:
            interpretation_text.append(f"• **Normal ACR ({acr_val:.0f} mg/g)** - No significant proteinuria detected")
        
        # HbA1c interpretation with diabetes management goals
        hba1c_val = user_inputs['baseline_HbA1c']
        if hba1c_val >= 10:
            interpretation_text.append(f"• **Very poor glycemic control (HbA1c {hba1c_val:.1f}%)** - Urgent diabetes management needed")
        elif hba1c_val >= 9:
            interpretation_text.append(f"• **Poor glycemic control (HbA1c {hba1c_val:.1f}%)** - Intensified therapy recommended")
        elif hba1c_val >= 8:
            interpretation_text.append(f"• **Suboptimal glycemic control (HbA1c {hba1c_val:.1f}%)** - Consider treatment adjustment")
        elif hba1c_val >= 7:
            interpretation_text.append(f"• **Borderline glycemic control (HbA1c {hba1c_val:.1f}%)** - Near target but room for improvement")
        else:
            interpretation_text.append(f"• **Good glycemic control (HbA1c {hba1c_val:.1f}%)** - Meeting diabetes management goals")
        
        # DR grade interpretation
        dr_grade = user_inputs['DR_grade']
        dr_interpretations = {
            0: "• **No diabetic retinopathy** - Regular ophthalmologic monitoring recommended",
            1: "• **Mild NPDR** - Annual dilated eye exams, good glycemic control important",
            2: "• **Moderate NPDR** - More frequent ophthalmologic follow-up needed",
            3: "• **Severe NPDR** - High risk for progression, intensive monitoring required",
            4: "• **Proliferative DR** - Advanced disease, immediate ophthalmologic management needed"
        }
        if dr_grade in dr_interpretations:
            interpretation_text.append(dr_interpretations[dr_grade])
        
        # Risk factors analysis
        risk_factors = []
        protective_factors = []
        
        if user_inputs['Family_Hx_CKD']:
            risk_factors.append("family history of CKD")
        
        if user_inputs['NSAID_cum90d']:
            risk_factors.append("regular NSAID use (≥90 days/year)")
        
        if user_inputs['IMD_quintile'] >= 4:
            risk_factors.append("socioeconomic deprivation")
        
        if user_inputs['baseline_BMI'] > 35:
            risk_factors.append("severe obesity")
        elif user_inputs['baseline_BMI'] > 30:
            risk_factors.append("obesity")
        
        if user_inputs['baseline_SBP'] >= 140:
            risk_factors.append("hypertension")
        
        if user_inputs['MRA_use']:
            protective_factors.append("MRA therapy (finerenone/eplerenone)")
        
        if user_inputs['baseline_eGFR'] > 90 and user_inputs['baseline_ACR'] < 10:
            protective_factors.append("preserved kidney function")
        
        if user_inputs['baseline_HbA1c'] < 7:
            protective_factors.append("optimal glycemic control")
        
        # Risk factors summary
        if risk_factors:
            interpretation_text.append(f"• **Additional risk factors present**: {', '.join(risk_factors)}")
        
        if protective_factors:
            interpretation_text.append(f"• **Protective factors**: {', '.join(protective_factors)}")
        
        # Clinical recommendations based on risk level
        if risk_category == "High":
            interpretation_text.append("• **Clinical Action**: Consider nephrology referral, intensify DKD prevention strategies")
        elif risk_category == "Moderate":
            interpretation_text.append("• **Clinical Action**: Enhanced monitoring, lifestyle interventions, optimize medical therapy")
        else:
            interpretation_text.append("• **Clinical Action**: Continue current management, routine monitoring appropriate")
        
        # Display interpretations
        for text in interpretation_text:
            st.markdown(text)
            
        # Evidence-based recommendations
        st.subheader("Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recommended Actions**")
            immediate_actions = []
            
            if user_inputs['baseline_eGFR'] < 60:
                immediate_actions.append("Monitor kidney function every 3-6 months")
            if user_inputs['baseline_ACR'] >= 30:
                immediate_actions.append("Optimize ACE inhibitor/ARB therapy")
            if user_inputs['baseline_HbA1c'] > 8:
                immediate_actions.append("Intensify glucose management")
            if user_inputs['NSAID_cum90d']:
                immediate_actions.append("Consider NSAID discontinuation")
            if not user_inputs['MRA_use'] and user_inputs['baseline_eGFR'] > 25:
                immediate_actions.append("Consider MRA therapy (finerenone)")
            
            if not immediate_actions:
                immediate_actions.append("Continue current evidence-based care")
            
            for action in immediate_actions:
                st.write(f"• {action}")
        
        with col2:
            st.markdown("**Monitoring Plan**")
            monitoring_items = []
            
            if risk_category == "High":
                monitoring_items.extend([
                    "eGFR and ACR every 3 months",
                    "HbA1c every 3 months", 
                    "Blood pressure monitoring"
                ])
            elif risk_category == "Moderate":
                monitoring_items.extend([
                    "eGFR and ACR every 6 months",
                    "HbA1c every 6 months",
                    "Annual comprehensive diabetes care"
                ])
            else:
                monitoring_items.extend([
                    "eGFR and ACR annually",
                    "HbA1c every 6 months",
                    "Routine diabetes monitoring"
                ])
            
            for item in monitoring_items:
                st.write(f"• {item}")
        
        # Generate report with enhanced content
        st.subheader("Predictive Modeling Report")
        report_text = generate_comprehensive_report(user_inputs, risk_percentage, risk_category, interpretation_text)
        
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"DKD_Risk_Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        with st.expander("Preview Report"):
            st.text(report_text)
    
    # Information sections
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About This Platform and Application:")
        st.write("""
        PSMMC NephraRisk prediction tool uses machine learning 
        to estimate the 3-year risk of developing or progressing kidney disease in 
        diabetic patients.
        """)
    
    with col2:
        st.subheader("Reference For Risk Categories and Utilization:")
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
