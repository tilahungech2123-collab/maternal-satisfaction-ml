import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# --- Load the saved model and preprocessing tools ---
try:
    loaded_model = joblib.load('logistic_regression_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    loaded_feature_columns = joblib.load('feature_columns.joblib')
    loaded_categorical_features_encoded = joblib.load('categorical_features_encoded.joblib')
    loaded_y_test = joblib.load('y_test.joblib')
    loaded_y_pred = joblib.load('y_pred.joblib')
    loaded_coef_importance = joblib.load('coef_importance.joblib')
    loaded_perm_importance = joblib.load('perm_importance.joblib')
except FileNotFoundError as e:
    st.error(f"Error: Missing model or preprocessing files: {e}")
    st.stop()

# --- Try Loading Raw Data for Preview and Plots ---
try:
    # Reading the primary dataset to show raw preview in Overview tab
    df_raw = pd.read_spss('/content/drive/MyDrive/Delivery care satisfaction.sav')
except Exception:
    df_raw = None

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Maternal Satisfaction")

# --- Display Top Cover Image of a Pregnant Mother from Google Drive ---
col_spacer1, col_img_center, col_spacer2 = st.columns([1, 2, 1])
with col_img_center:
    local_image_path = "/content/drive/MyDrive/pregnancy mother.jpg"
    if os.path.exists(local_image_path):
        st.image(local_image_path, caption="Maternal Care and Support", width=450)
    else:
        st.warning("Image not found at Google Drive path: /content/drive/MyDrive/pregnancy mother.jpg")

# --- Top Banner with Elegant Pregnant Mother Illustration and Title ---
col_img, col_title = st.columns([1, 5])
with col_img:
    st.image("https://img.icons8.com/illustrations/opacity/250/pregnant.png", width=150)
with col_title:
    st.markdown("# Maternal Satisfaction with Delivery Care — Bahir Dar, Ethiopia")
    st.markdown("### exploration of delivery care data and logistic-regression ML models for predicting maternal satisfaction")

# --- Navigation Tabs ---
tab_overview, tab_descriptive, tab_performance, tab_predict = st.tabs([
    "📊 Overview of datasets",
    "📈 Descriptive Charts",
    "🧠 Model Performance",
    "🔮 Predict"
])

# --- Helper function for preprocessing ---
def preprocess_new_data(new_raw_data, loaded_categorical_features_encoded, loaded_feature_columns, loaded_scaler):
    new_data_encoded = pd.get_dummies(new_raw_data, columns=loaded_categorical_features_encoded, drop_first=True)
    final_df_processed = pd.DataFrame(columns=loaded_feature_columns)
    final_df_processed = pd.concat([final_df_processed, new_data_encoded], ignore_index=True)
    final_df_processed = final_df_processed.fillna(0)
    final_df_processed = final_df_processed[loaded_feature_columns]
    numerical_cols_for_scaling_loaded = [col for col in loaded_feature_columns if col in loaded_scaler.feature_names_in_]
    final_df_processed[numerical_cols_for_scaling_loaded] = loaded_scaler.transform(final_df_processed[numerical_cols_for_scaling_loaded])
    return final_df_processed

# --- TAB 1: OVERVIEW ---
with tab_overview:
    st.header("Project Overview")

    # Overall Metrics
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric(label="Total Mothers Surveyed", value="894")
    with col_stat2:
        st.metric(label="Total Variables", value="46")
    with col_stat3:
        st.metric(label="Overall Delivery Care Satisfaction", value="61.5%")

    st.markdown("""
    This application provides an interactive portal to analyze and predict maternal satisfaction among vaginal and cesarean section delivery care services in Bahir Dar city health facilities, Northwest Ethiopia.

    ### Key Variables in Deployment:
    - **Socio-demographic Factors:** Age, Marital Status, Religion, Ethnicity, Education, Occupation, Income, Residence.
    - **Clinical & Obstetric History:** Gestational Age, Birth Weight, Parity, ANC Attendance, HIV Test Result, Mode of Delivery.
    - **Quality & Facility Perceptions:** Privacy at delivery, Waiting time at hospital, Distance to Clinic.
    """)

    # Raw Data Preview section
    st.subheader("📋 Raw Dataset Preview")
    if df_raw is not None:
        df_preview = df_raw.head(20).astype(str)
        st.dataframe(df_preview)
    else:
        st.info("Raw dataset preview currently unavailable in this deployment environment.")

# --- TAB 2: DESCRIPTIVE CHARTS ---
with tab_descriptive:
    st.header("Descriptive Charts & Insights")
    st.markdown("""
    Explore the baseline distributions and critical indicators collected from the 894 surveyed mothers.
    These plots reveal how key demographic, obstetric, and institutional factors correlate with maternal satisfaction.
    """)

    if df_raw is not None:
        st.subheader("Custom Categorical Variable Explorer")
        all_categorical_cols = df_raw.select_dtypes(include='category').columns.tolist()
        all_categorical_cols.sort()

        selected_var = st.selectbox("Choose a categorical variable to visualize:", all_categorical_cols, index=all_categorical_cols.index('modeofdeliveryrecode') if 'modeofdeliveryrecode' in all_categorical_cols else 0)

        col_exp_left, col_exp_right = st.columns([2, 1])
        with col_exp_left:
            fig_custom, ax_custom = plt.subplots(figsize=(8, 4.5))
            sns.countplot(data=df_raw, x=selected_var, hue='satisfaction', palette='viridis', ax=ax_custom)
            ax_custom.set_title(f"Distribution of {selected_var} by Satisfaction")
            ax_custom.set_xlabel(selected_var)
            ax_custom.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_custom)

        with col_exp_right:
            st.markdown(f"**Frequency Breakdown for `{selected_var}`**")
            freq_df = df_raw[selected_var].value_counts().reset_index()
            freq_df.columns = [selected_var, 'Count']
            st.dataframe(freq_df, use_container_width=True)

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Overall Distribution of Maternal Satisfaction")
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            sns.countplot(data=df_raw, x='satisfaction', palette='viridis', hue='satisfaction', legend=False, ax=ax1)
            ax1.set_title("Distribution of Maternal Satisfaction")
            ax1.set_xlabel("Satisfaction Level")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        with col_right:
            st.subheader("Satisfaction by Mode of Delivery")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            if 'modeofdeliveryrecode' in df_raw.columns:
                sns.countplot(data=df_raw, x='modeofdeliveryrecode', hue='satisfaction', palette='Set2', ax=ax2)
                ax2.set_title("Satisfaction by Mode of Delivery")
                ax2.set_xlabel("Mode of Delivery")
                ax2.set_ylabel("Count")
            else:
                ax2.text(0.5, 0.5, "'modeofdeliveryrecode' not found in dataset", ha='center', va='center')
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Top Qualitative Quality Determinants")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("**1. Cleanliness of Toilets / Wards**")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            if 'accessandcleanessoftoiletrecode' in df_raw.columns:
                sns.countplot(data=df_raw, x='accessandcleanessoftoiletrecode', hue='satisfaction', palette='coolwarm', ax=fig3.gca())
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right')
            st.pyplot(fig3)

        with col_f2:
            st.markdown("**2. Privacy During Delivery**")
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            if 'privacyrecode' in df_raw.columns:
                sns.countplot(data=df_raw, x='privacyrecode', hue='satisfaction', palette='coolwarm', ax=fig4.gca())
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30, ha='right')
            st.pyplot(fig4)

        with col_f3:
            st.markdown("**3. Waiting Time Perceptions**")
            fig5, ax5 = plt.subplots(figsize=(5, 4))
            if 'waitingtimerecode' in df_raw.columns:
                sns.countplot(data=df_raw, x='waitingtimerecode', hue='satisfaction', palette='coolwarm', ax=fig5.gca())
                ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right')
            st.pyplot(fig5)
    else:
        st.info("Detailed visual graphics require the raw dataset connection, which is currently unavailable.")

# --- TAB 3: MODEL PERFORMANCE ---
with tab_performance:
    st.header("Model Performance Metrics & Evaluation")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric(label="Accuracy", value="95.53%")
    with m_col2:
        st.metric(label="Precision", value="97.22%")
    with m_col3:
        st.metric(label="Recall", value="95.45%")
    with m_col4:
        st.metric(label="F1-Score", value="96.33%")
    with m_col5:
        st.metric(label="AUC-ROC", value="0.9859")
    with m_col6:
        st.metric(label="Brier Score", value="0.0404")

    st.markdown("---")
    col_perf1, col_perf2 = st.columns(2)

    with col_perf1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(loaded_y_test, loaded_y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Satisfied', 'Satisfied'],
                    yticklabels=['Not Satisfied', 'Satisfied'], ax=ax_cm)
        st.pyplot(fig_cm)

    with col_perf2:
        st.subheader("Feature Importances (Coefficients)")
        fig_coef, ax_coef = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=loaded_coef_importance.head(10).values, y=loaded_coef_importance.head(10).index, palette='coolwarm', ax=ax_coef)
        st.pyplot(fig_coef)

# --- TAB 4: PREDICT ---
with tab_predict:
    st.header("Predict Maternal Satisfaction")
    st.write("Enter details to predict whether a mother is likely to be satisfied with her delivery care experience.")

    with st.expander("Input Patient and Facility Characteristics", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 60, 25)
            marital_status = st.selectbox("Marital Status", ["married", "single", "divorced", "widowed"])
            gestational_age = st.slider("Gestational Age (weeks)", 20, 42, 38)
            delivery_mode = st.selectbox("Mode of Delivery", ["vaginal", "cesarean"])
            education = st.selectbox("Education Level", ["no formal education", "grade 1 to 4", "grade 5 to 8", "grade 9 to 10", "grade 11 to 12", "diploma and above", "unable to read and write"])
            reason = st.selectbox("Reason for delivery place", ["referred by health worker", "referred by family/friend", "near to home", "good quality of care", "other"])

        with col2:
            income = st.number_input("Monthly Income (ETB)", min_value=0, value=3000)
            religion = st.selectbox("Religion", ["orthodox", "muslim", "protestant", "catholic", "other"])
            birth_weight = st.number_input("Birth Weight (grams)", min_value=500, value=3000)
            anc_attend = st.selectbox("ANC Attendance", ["yes", "no"])
            occupation = st.selectbox("Occupation", ["housewife", "farmer", "merchant", "government employee", "other"])
            waiting_time = st.selectbox("Waiting Time at hospital", ["less than 30min", "30min-1hr", "1-2hr", "more than 2hr", "no wait"])

        with col3:
            dist_clinic = st.number_input("Distance to Clinic (km)", min_value=0.0, value=2.0)
            ethnicity = st.selectbox("Ethnicity", ["amhara", "oromo", "tigrie", "other"])
            parity = st.selectbox("Parity", ["nullipara", "para 1-2", "para 3-4", "para 5 and above"])
            hiv_result = st.selectbox("HIV Test Result", ["positive", "negative", "unknown"])
            residence = st.selectbox("Residence", ["rural", "urban"])
            privacy = st.selectbox("Privacy at delivery", ["satisfied", "somehow satisfied", "not satisfied"])

    new_raw_data = pd.DataFrame({
        'q101age': [float(age)],
        'q102mstatu': [marital_status],
        'q103religi': [religion],
        'q104ethini': [ethnicity],
        'q105educat': [education],
        'q106occupa': [occupation],
        'q107mincom': [float(income)],
        'q108reside': [residence],
        'q1091distk': [float(dist_clinic)],
        'q110reason': [reason],
        'q111timeta': ['less than 30min'],
        'q112waitpl': [waiting_time],
        'q113privac': [privacy],
        'q201parity': [parity],
        'q202cpplan': ['yes'],
        'q203ancatt': [anc_attend],
        'q204hivres': [hiv_result],
        'q205ga': [float(gestational_age)],
        'q206delimo': [delivery_mode],
        'q207nstatu': ['single'],
        'q208pdhsta': ['alive'],
        'q209nsex': ['female'],
        'q2010bw': [float(birth_weight)],
        'q301greeti': ['very good'],
        'q302respec': ['very good'],
        'q303delive': ['very good'],
        'q411owners': ['no'],
        'ageafterrecod': ['25-29'],
        'monthlyincomerecode': ['2001-3000'],
        'gestationaagerecode': ['37-41 week'],
        'birthweightofbaby': ['2500-4000'],
        'modeofdeliveryrecode': [delivery_mode],
        'waitingtimerecode': [waiting_time],
        'privacyrecode': [privacy],
        'encourageandsupportatdeliveryrecode': ['satisfied'],
        'politenessrecode': ['satisfied'],
        'availabilityofmedicalequipmentsrecode': ['satisfied'],
        'counselingrecode': ['satisfied'],
        'overallcleanessrecode': ['satisfied'],
        'accessandcleanessoftoiletrecode': ['satisfied'],
        'waitingareacleanessandconfort': ['satisfied'],
        'availabilityofbedintheward': ['satisfied'],
        'distancerecode1': ['<=10 kilo meters']
    }, index=[0])

    if st.button('Predict Satisfaction'):
        processed_input = preprocess_new_data(new_raw_data, loaded_categorical_features_encoded, loaded_feature_columns, loaded_scaler)
        prediction = loaded_model.predict(processed_input)
        prediction_proba = loaded_model.predict_proba(processed_input)[:, 1]
        predicted_satisfaction = 'Satisfied' if prediction[0] == 1 else 'Not Satisfied'

        st.subheader('Prediction Results:')
        st.write(f"Predicted Status: **{predicted_satisfaction}**")
        st.write(f"Confidence Level: **{prediction_proba[0]:.2%}**")
