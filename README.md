# Employee Salary Prediction – IBM SkillsBuild Internship

This repository contains the end‑to‑end machine learning project I completed as part of the **IBM SkillsBuild Internship Program**. The goal is to predict whether an employee’s annual income exceeds a threshold (e.g., `>50K`) using demographic and employment attributes. The project demonstrates the full ML lifecycle: data preprocessing, feature engineering, model training/evaluation, and a **Streamlit** web app for interactive inference.

---

## 📌 Project Objectives
- Ingest and clean a structured HR/“Adult” salary dataset.
- Transform raw categorical and numerical features into a machine‑learning–ready format.
- Train and compare multiple classification algorithms.
- Select the best model based on accuracy and classification metrics.
- Deploy the model via a lightweight **Streamlit** application.

---

## 🗂 Dataset
Based on a public “adult income” style dataset with columns such as:
`age`, `workclass`, `education`, `occupation`, `marital-status`, `gender`, `hours-per-week`, etc., plus the target label `income`.

### Key Preprocessing Steps
- Dropped low‑utility / duplicate columns: e.g., `fnlwgt`, `educational-num`, `capital-gain`, `capital-loss`, `native-country`, etc.
- Label encoding of categorical features (e.g., `workclass`, `education`, `occupation`).
- Feature scaling using **MinMaxScaler** / **StandardScaler** (especially important for distance‑based models).
- Train/test split with stratification to preserve class balance.

---

## 🧠 Models Trained
Implemented using **scikit-learn**:

| Model | Notes |
|-------|------|
| LogisticRegression | Fast linear baseline |
| RandomForestClassifier | Ensemble of decision trees; handles non-linearities |
| KNeighborsClassifier | Distance-based; benefits from scaling |
| SVC (Support Vector Classifier) | Kernel-based classification |
| GradientBoostingClassifier | Boosted weak learners for higher accuracy |

Each model was wrapped in a pipeline (`Scaler → Estimator`) for consistent preprocessing.

### Evaluation Metrics
- **Primary:** Accuracy  
- **Secondary:** Precision, Recall, F1-score (via `classification_report`)

### Best Model
The **GradientBoostingClassifier** achieved the highest test accuracy (≈**83.56**), with strong precision and recall across both income classes.  
This model was serialized using `joblib` for deployment as `best_model.pkl`.

<img width="1520" height="1340" alt="image" src="https://github.com/user-attachments/assets/f00aa4a5-1a4a-4e76-90c2-d5dfb27c88b5" />


---

## 📊 Visualizations
Exploratory and model diagnostics (Matplotlib/Seaborn):
- Boxplots for outlier detection (e.g., `age`)
- Correlation/feature distributions
- Model accuracy comparison bar chart

---

## 🚀 Streamlit App
A user-friendly interface (`app.py`) allows real-time predictions.

The app URL is given as follows-

https://employee-salary-prediction-yujqsax5icqva2kgwkwnrj.streamlit.app/

<img width="2936" height="1664" alt="image" src="https://github.com/user-attachments/assets/899c830e-b514-47f7-8f3a-ffa0589f0af0" />


### Features
- Input form for employee attributes (dropdowns/text fields)
- Preprocessing mirrors training pipeline (same encoders/scalers)
- Loads the persisted `best_model.pkl` to generate predictions
- Outputs predicted salary class and (optionally) probability

### Run Locally

pip install -r requirements.txt

streamlit run app.py

#📁 Repository Structure
├── app.py                           # Streamlit inference app

├── employee salary prediction.ipynb # Development / training notebook

├── adult 3 (1).csv                  # Dataset

├── best_model.pkl                   # Saved best-performing model

├── requirements.txt                 # Python dependencies

└── README.md

