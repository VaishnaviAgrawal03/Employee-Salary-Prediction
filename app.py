import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Manual encoding maps
workclass_map = {
    "Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2,
    "Federal-gov": 3, "Local-gov": 4, "State-gov": 5,
    "Without-pay": 6, "Never-worked": 7
}

education_map = {
    "Bachelors": 0,
    "Some-college": 1,
    "11th": 2,
    "HS-grad": 3,
    "Prof-school": 4,
    "Assoc-acdm": 5,
    "Assoc-voc": 6,
    "9th": 7,
    "7th-8th": 8,
    "12th": 9,
    "Masters": 10,
    "1st-4th": 11,
    "10th": 12,
    "Doctorate": 13,
    "5th-6th": 14,
    "Preschool": 15
}

marital_status_map = {
    "Never-married": 0, "Married-civ-spouse": 1, "Divorced": 2,
    "Separated": 3, "Married-spouse-absent": 4, "Widowed": 5
}
occupation_map = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
    "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
    "Transport-moving": 10, "Priv-house-serv": 11,
    "Protective-serv": 12, "Armed-Forces": 13
}
gender_map = {"Male": 1, "Female": 0}

# Streamlit UI
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Input collection
age = st.sidebar.slider("Age", 18, 65, 30)
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
education = st.sidebar.selectbox("Education", list(education_map.keys()))
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# readable dataframe for UI display
display_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Build input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass_map[workclass]],
    'education': [education_map[education]],
    'marital-status': [marital_status_map[marital_status]],
    'occupation': [occupation_map[occupation]],
    'gender': [gender_map[gender]],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Show readable input
st.write("### 🔍 Input Data")
st.write(display_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)

    # Format prediction nicely
    if prediction[0] == "<=50K":
        st.success("✅ Prediction: Estimated Salary is  ≤ 50,000")
    else:
        st.success("✅ Prediction: Estimated Salary is  > 50,000")

# Batch Prediction Section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a  CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded data preview:", batch_data.head())

    try:
        # Map categorical columns
        categorical_cols = ['workclass', 'marital-status', 'occupation', 'gender']
        mapping_dicts = {
            'workclass': workclass_map,
            'education': education_map,
            'marital-status': marital_status_map,
            'occupation': occupation_map,
            'gender': gender_map
        }

        for col in categorical_cols:
            batch_data[col] = batch_data[col].map(mapping_dicts[col])
            # Fill any unmapped category with -1
            batch_data[col].fillna(-1, inplace=True)

        # Handle any remaining NaN values
        for col in batch_data.columns:
            if batch_data[col].isnull().any():
                if batch_data[col].dtype == 'object':
                    batch_data[col].fillna(batch_data[col].mode()[0], inplace=True)
                else:
                    batch_data[col].fillna(batch_data[col].mean(), inplace=True)

        # Drop label column if present (for pure prediction)
        if 'income' in batch_data.columns:
            batch_data.drop(columns=['income'], inplace=True)

        # Predict
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds

        st.write("✅ Predictions:")
        st.write(batch_data.head())

        # Download CSV
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error in batch prediction: {e}")
