"""
Student Performance Prototyping App
To run: streamlit run student_performance_app.py
Models needed: best_regression_model.pkl, best_classification_model.pkl
Upload format: CSV with same columns as training data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Student Performance AI Prototype",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #7c3aed;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stSidebar {
        background-color: #7c3aed;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Load Models & Recreate Preprocessor ---
@st.cache_resource
def load_assets():
    try:
        reg_model = joblib.load('best_regression_model.pkl')
        clf_model = joblib.load('best_classification_model.pkl')
        return reg_model, clf_model, True
    except Exception as e:
        return None, None, False

best_reg, best_clf, MODELS_LOADED = load_assets()

# Constants for preprocessing (Must match training data)
NUMERIC_COLS = ['age', 'weeks_in_course', 'hours_spent_learning_per_week', 
                'practice_problems_solved', 'projects_completed', 
                'tutorial_videos_watched', 'uses_kaggle', 
                'participates_in_discussion_forums', 'debugging_sessions_per_week', 
                'self_reported_confidence_python', 'total_learning_effort']

CATEGORICAL_COLS = ['country', 'prior_programming_experience']

# --- 3. Sidebar Inputs ---
st.sidebar.title("🎓 Student Attributes")
st.sidebar.markdown("Input data for a single student prediction.")

def get_user_input():
    # Split into columns for a cleaner layout
    col1 = st.sidebar
    
    country = col1.selectbox("Country", ["Pakistan", "Nigeria", "India", "UK", "USA", "Other"])
    experience = col1.selectbox("Prior Experience", ["None", "Beginner", "Intermediate", "Advanced"])
    age = col1.slider("Age", 15, 75, 25)
    weeks = col1.slider("Weeks in Course", 1, 20, 10)
    hours = col1.slider("Hours/Week", 0.0, 40.0, 10.0)
    problems = col1.number_input("Practice Problems Solved", 0, 1000, 50)
    projects = col1.slider("Projects Completed", 0, 10, 1)
    videos = col1.number_input("Tutorial Videos Watched", 0, 500, 20)
    kaggle = col1.checkbox("Uses Kaggle", value=False)
    forums = col1.checkbox("Participates in Forums", value=False)
    debugging = col1.slider("Debugging Sessions/Week", 0, 20, 3)
    confidence = st.sidebar.select_slider("Python Confidence", options=list(range(1, 11)), value=5)
    
    # Feature Engineering (Reuse logic)
    total_effort = weeks * hours
    
    data = {
        'age': age,
        'country': country,
        'prior_programming_experience': experience,
        'weeks_in_course': weeks,
        'hours_spent_learning_per_week': hours,
        'practice_problems_solved': problems,
        'projects_completed': projects,
        'tutorial_videos_watched': videos,
        'uses_kaggle': 1 if kaggle else 0,
        'participates_in_discussion_forums': 1 if forums else 0,
        'debugging_sessions_per_week': debugging,
        'self_reported_confidence_python': confidence,
        'total_learning_effort': total_effort
    }
    return pd.DataFrame([data])

# --- 4. Main App UI ---
st.title("🎓 Student Exam Performance Predictor")
st.markdown("""
This prototype app uses optimized Machine Learning models to predict a student's final exam performance 
based on their behavior and background.
""")

if not MODELS_LOADED:
    st.error("⚠️ SAVED MODELS NOT FOUND! Please run the training notebooks first.")
    st.info("The app requires `best_regression_model.pkl` and `best_classification_model.pkl` to function.")
    st.stop()

tabs = st.tabs(["🎯 Single Prediction", "📂 Batch Processing", "📊 Model Insights"])

# --- Tab 1: Single Prediction ---
with tabs[0]:
    user_data = get_user_input()
    
    st.subheader("Current Student Profile")
    st.dataframe(user_data, use_container_width=True)
    
    if st.button("🚀 Run Prediction", type="primary"):
        # Regression prediction
        reg_pred = best_reg.predict(user_data)[0]
        
        # Classification prediction (Probability)
        clf_probs = best_clf.predict_proba(user_data)[0]
        pass_prob = clf_probs[1]
        
        # Display Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("Predicted Exam Score", f"{reg_pred:.1f}%")
        
        with m_col2:
            st.metric("Pass Probability", f"{pass_prob*100:.1f}%")
            
        with m_col3:
            status = "PASS ✅" if pass_prob >= 0.5 else "FAIL ❌"
            color = "green" if pass_prob >= 0.5 else "red"
            st.markdown(f"**Final Status**: <span style='color:{color}; font-size:24px; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

        # Log prediction
        log_entry = user_data.copy()
        log_entry['predicted_score'] = reg_pred
        log_entry['pass_probability'] = pass_prob
        log_entry['timestamp'] = datetime.datetime.now()
        
        try:
            log_df = pd.read_csv('predictions_log.csv')
            log_df = pd.concat([log_df, log_entry], ignore_index=True)
        except:
            log_df = log_entry
        log_df.to_csv('predictions_log.csv', index=False)

# --- Tab 2: Batch Processing ---
with tabs[1]:
    st.subheader("Bulk Analysis")
    uploaded_file = st.file_uploader("Upload Student Data (CSV)", type="csv")
    
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_cols = ['age', 'country', 'prior_programming_experience', 'weeks_in_course', 
                         'hours_spent_learning_per_week', 'practice_problems_solved', 
                         'projects_completed', 'tutorial_videos_watched', 'uses_kaggle', 
                         'participates_in_discussion_forums', 'debugging_sessions_per_week', 
                         'self_reported_confidence_python']
        
        missing = [c for c in required_cols if c not in df_batch.columns]
        
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.success("File validated successfully!")
            
            # Feature Engineering for batch
            df_batch['total_learning_effort'] = df_batch['weeks_in_course'] * df_batch['hours_spent_learning_per_week']
            
            # Predict
            reg_preds = best_reg.predict(df_batch)
            clf_probs = best_clf.predict_proba(df_batch)[:, 1]
            
            df_batch['Predicted_Score'] = reg_preds
            df_batch['Pass_Probability'] = clf_probs
            df_batch['Recommendation'] = ["Support Required" if p < 0.5 else "On Track" for p in clf_probs]
            
            st.divider()
            st.dataframe(df_batch, use_container_width=True)
            
            # Summary Metrics for Batch
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("Avg Predicted Score", f"{df_batch['Predicted_Score'].mean():.1f}%")
            b_col2.metric("Projected Pass Rate", f"{(df_batch['Pass_Probability'] >= 0.5).mean()*100:.1f}%")
            
            # Visualization for batch
            fig = px.pie(df_batch, names='Recommendation', title='Batch Summary: Pass/Fail Breakdown', 
                         color_discrete_map={'On Track': '#2ecc71', 'Support Required': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv, file_name="student_predictions.csv", mime="text/csv")

# --- Tab 3: Model Insights ---
with tabs[2]:
    st.subheader("Explainable AI (XAI)")
    
    # Feature Importance visualization
    # Note: Extracting from Best GB Regressor (best_reg)
    try:
        # Get model components
        model_part = best_reg.named_steps['model']
        prep_part = best_reg.named_steps['prep']
        
        # Get feature names after one-hot encoding
        ohe_cols = list(prep_part.transformers_[1][1].get_feature_names_out(CATEGORICAL_COLS))
        all_features = list(NUMERIC_COLS) + ohe_cols
        
        # Get importances
        importances = model_part.feature_importances_
        imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values('Importance', ascending=True).tail(10)
        
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', 
                         title="Top 10 Drivers of Exam Performance",
                         color='Importance', color_continuous_scale='Teal')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.info("**Insight**: Total Learning Effort and Practice Problems are weighted most heavily in our champion model.")
        
    except Exception as e:
        st.write("Feature importance visualization is only available for tree-based models.")

    # Model Performance Stats (Hardcoded from notebook)
    st.divider()
    st.subheader("Champion Model Performance")
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.markdown("**Regression (Exam Score Prediction)**")
        st.markdown("- Model: Tuned Gradient Boosting")
        st.markdown("- R² Score: ~0.87")
        st.markdown("- RMSE: ~7.2")
        
    with stats_col2:
        st.markdown("**Classification (Pass/Fail Prediction)**")
        st.markdown("- Model: Tuned Random Forest")
        st.markdown("- F1-Score: ~0.91")
        st.markdown("- Accuracy: ~92%")

st.sidebar.divider()
st.sidebar.caption("Built with ❤️ for Educational Analytics Prototype")
