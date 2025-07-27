import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

# Manual encoding maps
workclass_map = {
    "Federal-gov": 0, "Local-gov": 1, "Others": 2,
    "Private": 3, "Self-emp-inc": 4, "Self-emp-not-inc": 5,
    "State-gov": 6
}
education_map = {
    "10th": 0, "11th": 1, "12th": 2, "9th": 3, "Assoc-acdm": 4,
    "Assoc-voc": 5, "Bachelors": 6, "Doctorate": 7,
    "HS-grad": 8, "masters": 9, "Prof-school": 10,
    "Some-college": 11
}
marital_status_map = {
    "Divorced": 0, "Married-AF-spouse": 1, "Married-civ-spouse": 2,
    "Married-spouse-absent": 3, "Never-married": 4,
    "Separated": 5, "Widowed": 6
}
occupation_map = {
    "Adm-clerical": 0, "Armed-Forces": 1, "Craft-repair": 2,
    "Exec-managerial": 3, "Farming-fishing": 4, "Handlers-cleaners": 5,
    "Machine-op-inspct": 6, "Other-service": 7, "Others": 8,
    "Priv-house-serv": 9, "Prof-specialty": 10, "Protective-serv": 11,
    "Sales": 12, "Tech-support": 13, "Transport-moving": 14
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
