import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Multimodal Disease Prediction", page_icon="🩺", layout="wide")

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ----------------------------
# HELPER: load model + scaler
# ----------------------------
@st.cache_resource
def load_model_scaler(name):
    model_path = os.path.join(OUTPUTS_DIR, f"{name}_model.pkl")
    scaler_path = os.path.join(OUTPUTS_DIR, f"{name}_scaler.pkl")
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler


def predict(name, features: list):
    """features must be in the exact order used during training."""
    model, scaler = load_model_scaler(name)
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
    return pred, proba


def show_result(disease_name, pred, proba, positive_label=1, positive_text="Positive / At Risk",
                 negative_text="Negative / Low Risk"):
    st.markdown("---")
    is_positive = (pred == positive_label) or (str(pred) == str(positive_label))
    if is_positive:
        st.error(f"⚠️ **Result: {positive_text}** for {disease_name}")
    else:
        st.success(f"✅ **Result: {negative_text}** for {disease_name}")

    if proba is not None:
        confidence = max(proba) * 100
        st.write(f"Model confidence: **{confidence:.2f}%**")
    st.caption("⚠️ This is an ML prediction for educational purposes only, not a medical diagnosis. Please consult a doctor for actual medical advice.")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🩺 Disease Prediction")
st.sidebar.markdown("Select a disease to predict:")

disease = st.sidebar.radio(
    "Choose Disease",
    [
        "🏠 Home",
        "Diabetes",
        "Heart Disease",
        "Kidney Disease",
        "Liver Disease",
        "Stroke",
        "Hypertension",
        "Thyroid",
        "Breast Cancer",
        "COVID-19",
        "Lung Cancer",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Multimodal Disease Prediction System — Final Year Project")


# ============================================================
# HOME PAGE
# ============================================================
if disease == "🏠 Home":
    st.title("🩺 Multimodal Disease Prediction System")
    st.markdown("""
    Welcome! This app predicts the risk of **10 different diseases** using machine learning
    models trained on real medical datasets.

    👈 **Select a disease from the sidebar** to get started.

    ### Diseases Covered
    | Disease | Input Type |
    |---|---|
    | Diabetes | Lab values |
    | Heart Disease | Clinical values |
    | Kidney Disease | Lab values |
    | Liver Disease | Lab values |
    | Stroke | Lifestyle + clinical |
    | Hypertension | Lifestyle + clinical |
    | Thyroid | Lab values |
    | Breast Cancer | Cell nuclei measurements |
    | COVID-19 | Symptoms checklist |
    | Lung Cancer | Symptoms checklist |
    """)
    st.info("This tool is for educational/demo purposes only and is not a substitute for professional medical diagnosis.")


# ============================================================
# 1. DIABETES
# ============================================================
elif disease == "Diabetes":
    st.title("🩸 Diabetes Prediction")
    st.write("Enter the following lab values:")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes", type="primary"):
        features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        pred, proba = predict("diabetes", features)
        show_result("Diabetes", pred, proba)


# ============================================================
# 2. HEART DISEASE
# ============================================================
elif disease == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    st.write("Enter the following clinical values:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: typical angina, 1: atypical, 2: non-anginal, 3: asymptomatic")
        trestbps = st.number_input("Resting Blood Pressure", 0, 250, 120)
    with col2:
        chol = st.number_input("Cholesterol", 0, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 0, 250, 150)
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    if st.button("Predict Heart Disease", type="primary"):
        features = [
            age, 1 if sex == "Male" else 0, cp, trestbps, chol,
            1 if fbs == "Yes" else 0, restecg, thalach,
            1 if exang == "Yes" else 0, oldpeak, slope, ca, thal,
        ]
        pred, proba = predict("heart", features)
        show_result("Heart Disease", pred, proba)


# ============================================================
# 3. KIDNEY DISEASE
# ============================================================
elif disease == "Kidney Disease":
    st.title("🫘 Kidney Disease Prediction")
    st.write("Enter the following lab values:")
    st.caption("Note: 'id' column from training is excluded; a placeholder of 0 is used internally.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        bp = st.number_input("Blood Pressure", 0, 200, 80)
        sg = st.number_input("Specific Gravity", 1.0, 1.05, 1.02, format="%.3f")
        al = st.number_input("Albumin", 0, 5, 0)
        su = st.number_input("Sugar", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
        ba = st.selectbox("Bacteria", ["notpresent", "present"])
    with col2:
        bgr = st.number_input("Blood Glucose Random", 0, 500, 120)
        bu = st.number_input("Blood Urea", 0, 300, 40)
        sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)
        sod = st.number_input("Sodium", 0, 200, 140)
        pot = st.number_input("Potassium", 0.0, 15.0, 4.5)
        hemo = st.number_input("Hemoglobin", 0.0, 20.0, 13.0)
        pcv = st.number_input("Packed Cell Volume", 0, 60, 40)
        wc = st.number_input("White Blood Cell Count", 0, 20000, 8000)
    with col3:
        rc = st.number_input("Red Blood Cell Count", 0.0, 10.0, 5.0)
        htn = st.selectbox("Hypertension", ["no", "yes"])
        dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
        cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["no", "yes"])
        ane = st.selectbox("Anemia", ["no", "yes"])

    if st.button("Predict Kidney Disease", type="primary"):
        # Label encoding replicates sklearn's LabelEncoder alphabetical order
        def enc(val, options):
            return sorted(options).index(val)

        features = [
            0,  # placeholder for 'id' column from training
            age, bp, sg, al, su,
            enc(rbc, ["normal", "abnormal"]),
            enc(pc, ["normal", "abnormal"]),
            enc(pcc, ["notpresent", "present"]),
            enc(ba, ["notpresent", "present"]),
            bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
            enc(htn, ["no", "yes"]),
            enc(dm, ["no", "yes"]),
            enc(cad, ["no", "yes"]),
            enc(appet, ["good", "poor"]),
            enc(pe, ["no", "yes"]),
            enc(ane, ["no", "yes"]),
        ]
        pred, proba = predict("kidney", features)
        show_result("Kidney Disease", pred, proba)


# ============================================================
# 4. LIVER DISEASE
# ============================================================
elif disease == "Liver Disease":
    st.title("🫀 Liver Disease Prediction")
    st.write("Enter the following lab values:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tb = st.number_input("Total Bilirubin", 0.0, 50.0, 1.0)
        db = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.3)
        alk = st.number_input("Alkaline Phosphotase", 0, 2000, 200)
    with col2:
        alt = st.number_input("Alamine Aminotransferase", 0, 2000, 30)
        ast = st.number_input("Aspartate Aminotransferase", 0, 2000, 30)
        tp = st.number_input("Total Proteins", 0.0, 15.0, 6.5)
        alb = st.number_input("Albumin", 0.0, 10.0, 3.5)
        agr = st.number_input("Albumin and Globulin Ratio", 0.0, 5.0, 1.0)

    if st.button("Predict Liver Disease", type="primary"):
        features = [
            age, 1 if gender == "Male" else 0, tb, db, alk, alt, ast, tp, alb, agr,
        ]
        pred, proba = predict("liver", features)
        show_result("Liver Disease", pred, proba)


# ============================================================
# 5. STROKE
# ============================================================
elif disease == "Stroke":
    st.title("🧠 Stroke Prediction")
    st.write("Enter the following details:")
    st.caption("Note: 'id' column from training is excluded; a placeholder of 0 is used internally.")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 1, 120, 45)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    with col2:
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        glucose = st.number_input("Average Glucose Level", 0.0, 400.0, 100.0)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    if st.button("Predict Stroke", type="primary"):
        def enc(val, options):
            return sorted(options).index(val)

        features = [
            0,  # placeholder for 'id'
            enc(gender, ["Male", "Female", "Other"]),
            age,
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            enc(ever_married, ["No", "Yes"]),
            enc(work_type, ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]),
            enc(residence, ["Urban", "Rural"]),
            glucose, bmi,
            enc(smoking, ["never smoked", "formerly smoked", "smokes", "Unknown"]),
        ]
        pred, proba = predict("stroke", features)
        show_result("Stroke", pred, proba)


# ============================================================
# 6. HYPERTENSION
# ============================================================
elif disease == "Hypertension":
    st.title("💓 Hypertension Prediction")
    st.write("Enter the following lifestyle and clinical details:")
    st.caption("Note: 'Country' from training is excluded; a placeholder of 0 is used internally.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 40)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        chol = st.number_input("Cholesterol", 0, 500, 200)
        sbp = st.number_input("Systolic BP", 0, 250, 120)
        dbp = st.number_input("Diastolic BP", 0, 200, 80)
        smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "High"])
    with col2:
        activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        family = st.selectbox("Family History", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        salt = st.selectbox("Salt Intake", ["Low", "Moderate", "High"])
        sleep = st.number_input("Sleep Duration (hrs)", 0.0, 14.0, 7.0)
        hr = st.number_input("Heart Rate", 0, 200, 75)
    with col3:
        ldl = st.number_input("LDL", 0, 300, 100)
        hdl = st.number_input("HDL", 0, 150, 50)
        trig = st.number_input("Triglycerides", 0, 600, 150)
        glucose = st.number_input("Glucose", 0, 400, 100)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary"])
        employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"])

    if st.button("Predict Hypertension", type="primary"):
        def enc(val, options):
            return sorted(options).index(val)

        features = [
            0,  # placeholder for 'Country'
            age, bmi, chol, sbp, dbp,
            enc(smoking, ["Non-Smoker", "Smoker"]),
            enc(alcohol, ["None", "Moderate", "High"]),
            enc(activity, ["Low", "Moderate", "High"]),
            enc(family, ["No", "Yes"]),
            enc(diabetes, ["No", "Yes"]),
            enc(stress, ["Low", "Moderate", "High"]),
            enc(salt, ["Low", "Moderate", "High"]),
            sleep, hr, ldl, hdl, trig, glucose,
            enc(gender, ["Male", "Female"]),
            enc(education, ["Primary", "Secondary", "Tertiary"]),
            enc(employment, ["Employed", "Unemployed", "Retired"]),
        ]
        pred, proba = predict("hypertension", features)
        # Target was text "High"/"Low" -> LabelEncoder sorted alphabetically: High=0, Low=1
        show_result("Hypertension", pred, proba, positive_label=0,
                     positive_text="High Risk", negative_text="Low Risk")


# ============================================================
# 7. THYROID
# ============================================================
elif disease == "Thyroid":
    st.title("🦋 Thyroid Disease Prediction")
    st.write("Enter the following details:")

    yn = ["f", "t"]  # false/true as in original dataset

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 40)
        sex = st.selectbox("Sex", ["F", "M"])
        on_thyroxine = st.selectbox("On Thyroxine", yn)
        query_thyroxine = st.selectbox("Query On Thyroxine", yn)
        on_antithyroid = st.selectbox("On Antithyroid Medication", yn)
        sick = st.selectbox("Sick", yn)
        pregnant = st.selectbox("Pregnant", yn)
        thyroid_surgery = st.selectbox("Thyroid Surgery", yn)
        i131 = st.selectbox("I131 Treatment", yn)
        query_hypo = st.selectbox("Query Hypothyroid", yn)
    with col2:
        query_hyper = st.selectbox("Query Hyperthyroid", yn)
        lithium = st.selectbox("Lithium", yn)
        goitre = st.selectbox("Goitre", yn)
        tumor = st.selectbox("Tumor", yn)
        hypopituitary = st.selectbox("Hypopituitary", yn)
        psych = st.selectbox("Psych", yn)
        tsh_measured = st.selectbox("TSH Measured", yn)
        tsh = st.number_input("TSH", 0.0, 100.0, 2.0)
        t3_measured = st.selectbox("T3 Measured", yn)
        t3 = st.number_input("T3", 0.0, 20.0, 2.0)
    with col3:
        tt4_measured = st.selectbox("TT4 Measured", yn)
        tt4 = st.number_input("TT4", 0.0, 300.0, 100.0)
        t4u_measured = st.selectbox("T4U Measured", yn)
        t4u = st.number_input("T4U", 0.0, 3.0, 1.0)
        fti_measured = st.selectbox("FTI Measured", yn)
        fti = st.number_input("FTI", 0.0, 300.0, 100.0)
        tbg_measured = st.selectbox("TBG Measured", yn)
        tbg = st.number_input("TBG", 0.0, 100.0, 20.0)
        referral = st.selectbox("Referral Source", ["SVHC", "other", "SVI", "STMW", "SVHD"])

    if st.button("Predict Thyroid Disease", type="primary"):
        def enc(val, options):
            return sorted(options).index(val)

        features = [
            age, enc(sex, ["F", "M"]),
            enc(on_thyroxine, yn), enc(query_thyroxine, yn), enc(on_antithyroid, yn),
            enc(sick, yn), enc(pregnant, yn), enc(thyroid_surgery, yn), enc(i131, yn),
            enc(query_hypo, yn), enc(query_hyper, yn), enc(lithium, yn), enc(goitre, yn),
            enc(tumor, yn), enc(hypopituitary, yn), enc(psych, yn),
            enc(tsh_measured, yn), tsh,
            enc(t3_measured, yn), t3,
            enc(tt4_measured, yn), tt4,
            enc(t4u_measured, yn), t4u,
            enc(fti_measured, yn), fti,
            enc(tbg_measured, yn), tbg,
            enc(referral, ["SVHC", "other", "SVI", "STMW", "SVHD"]),
        ]
        pred, proba = predict("thyroid", features)
        # Target was text "P"/"N" -> LabelEncoder sorted alphabetically: N=0, P=1
        show_result("Thyroid Disease", pred, proba, positive_label=1,
                     positive_text="Positive (thyroid condition detected)",
                     negative_text="Negative (no thyroid condition detected)")


# ============================================================
# 8. BREAST CANCER
# ============================================================
elif disease == "Breast Cancer":
    st.title("🎗️ Breast Cancer Prediction")
    st.write("Enter the cell nuclei measurements:")
    st.caption("These values typically come from a digitized image of a fine needle aspirate (FNA) of breast tissue.")

    fields = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]

    values = {}
    cols = st.columns(3)
    for i, field in enumerate(fields):
        with cols[i % 3]:
            values[field] = st.number_input(field.replace("_", " ").title(), 0.0, 5000.0, 1.0, key=field)

    if st.button("Predict Breast Cancer", type="primary"):
        features = [values[f] for f in fields]
        pred, proba = predict("cancer", features)
        # diagnosis encoded alphabetically: B=0 (benign), M=1 (malignant)
        show_result("Breast Cancer", pred, proba, positive_label=1,
                     positive_text="Malignant (cancerous)", negative_text="Benign (non-cancerous)")


# ============================================================
# 9. COVID-19
# ============================================================
elif disease == "COVID-19":
    st.title("😷 COVID-19 Prediction")
    st.write("Select the symptoms/factors that apply:")

    symptoms = [
        "Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Running Nose",
        "Asthma", "Chronic Lung Disease", "Headache", "Heart Disease", "Diabetes",
        "Hyper Tension", "Fatigue ", "Gastrointestinal ", "Abroad travel",
        "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places",
        "Family working in Public Exposed Places", "Wearing Masks", "Sanitization from Market",
    ]

    values = {}
    cols = st.columns(4)
    for i, symptom in enumerate(symptoms):
        with cols[i % 4]:
            values[symptom] = st.checkbox(symptom.strip(), key=f"covid_{i}")

    if st.button("Predict COVID-19", type="primary"):
        # Yes/No encoded alphabetically -> No=0, Yes=1
        features = [1 if values[s] else 0 for s in symptoms]
        pred, proba = predict("covid", features)
        show_result("COVID-19", pred, proba, positive_label=1,
                     positive_text="Likely COVID-19 Positive", negative_text="Likely COVID-19 Negative")


# ============================================================
# 10. LUNG CANCER
# ============================================================
elif disease == "Lung Cancer":
    st.title("🫁 Lung Cancer Prediction")
    st.write("Enter the following details:")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Age", 1, 120, 45)
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        chronic = st.selectbox("Chronic Disease", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    with col2:
        allergy = st.selectbox("Allergy", ["No", "Yes"])
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
        coughing = st.selectbox("Coughing", ["No", "Yes"])
        sob = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    if st.button("Predict Lung Cancer", type="primary"):
        def yn(val):
            return 2 if val == "Yes" else 1  # original dataset uses 1=No, 2=Yes

        features = [
            1 if gender == "M" else 0,
            age,
            yn(smoking), yn(yellow_fingers), yn(anxiety), yn(peer_pressure),
            yn(chronic), yn(fatigue), yn(allergy), yn(wheezing),
            yn(alcohol), yn(coughing), yn(sob), yn(swallowing), yn(chest_pain),
        ]
        pred, proba = predict("lungcancer", features)
        # Target was text "YES"/"NO" -> LabelEncoder sorted alphabetically: NO=0, YES=1
        show_result("Lung Cancer", pred, proba, positive_label=1)
