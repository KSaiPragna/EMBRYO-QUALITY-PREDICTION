# 🧬 Embryo Quality Prediction Using PGx - AI & ML

## 📌 Project Overview

This project focuses on predicting embryo quality using AI and Machine Learning techniques based on PGx (Pharmacogenomics) and fertility-related clinical parameters.

The system performs:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multi-Model Machine Learning
- SHAP Explainability
- Streamlit Dashboard Deployment
- Embryo Quality Prediction

The project aims to assist fertility research and IVF-related decision support systems using data-driven predictive analytics.

---

# Dataset Availability 

The project uses a publicly available PCOS/Fertility-related dataset containing hormonal, biological, and fertility parameters.

Users may use similar fertility or PCOS-related datasets from Kaggle.

---

# 🔎 Suggested Kaggle Search Keywords

Search on Kaggle using:

```bash
PCOS Dataset
```

or

```bash
Fertility Prediction Dataset
```

or

```bash
IVF Embryo Dataset
```

or

```bash
Women's Health Fertility Dataset
```

---

# 🎯 Project Objectives

- Analyze fertility-related clinical parameters
- Predict embryo quality using machine learning
- Compare multiple ML classification models
- Visualize feature relationships and distributions
- Provide AI-assisted fertility insights
- Build an interactive medical prediction dashboard

---

# 📂 Project Structure

```bash
EMBRYO_QUALITY_PREDICTION/
│
├── eda_preprocessing.py
├── model_training.py
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── fertility_dataset.csv
│
├── models/
│   ├── XGBModel.pkl
│   ├── EQP_label_encoder.pkl
│   └── Scaler_EQP.pkl
│
├── outputs/
│   ├── cleaned_dataset.csv
│   ├── accuracy_comparison.png
│   ├── f1_comparison.png
│   └── shap_summary.png
│
└── images/
    └── dashboard_preview.png
```

---

# 📊 Dataset Features

The project uses the following fertility and PGx-related clinical features:

| Feature Name | Description |
|---|---|
| FSH(mIU/mL) | Follicle Stimulating Hormone |
| LH(mIU/mL) | Luteinizing Hormone |
| Age (yrs) | Patient age |
| AMH(ng/mL) | Anti-Müllerian Hormone |
| BMI | Body Mass Index |
| AFC | Antral Follicle Count |

---

# 🧪 Feature Engineering

The project creates:

```python
AFC = Follicle No. (L) + Follicle No. (R)
```

This feature is used as an important fertility indicator.

---

# 🎯 Target Variable

The machine learning model predicts:

```text
Embryo_Quality
```

---

# 📌 Target Classes

| Class | Meaning |
|---|---|
| Best | High embryo quality |
| Fair | Moderate embryo quality |
| Poor | Low embryo quality |

---

# ⚠️ Important Dataset Note

The target variable is generated using AFC values.

Example logic used:

```python
def classify_afc(afc):
    if afc <= 8:
        return 'Poor'
    elif 9 <= afc <= 12:
        return 'Fair'
    else:
        return 'Best'
```

Users may customize target thresholds based on medical or research requirements.

---

# 🔍 Exploratory Data Analysis (EDA)

The project performs:

- Distribution Analysis
- Histograms
- Boxplots
- Pairplots
- Correlation Heatmaps
- Skewness Analysis
- Q-Q Plots
- Outlier Detection

---

# 🤖 Machine Learning Models Used

The project compares multiple classification algorithms:

- XGBoost Classifier
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Gradient Boosting Classifier

---

# 🧠 Explainable AI (XAI)

The project uses:

```text
SHAP (SHapley Additive Explanations)
```

for feature importance and explainability analysis.

This helps understand:
- Important fertility factors
- Model decision behavior
- Clinical feature impact

---

# 📈 Model Evaluation Metrics

Models are evaluated using:

- Accuracy Score
- F1 Macro Score
- Confusion Matrix
- Classification Report

---

# ⚙️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib
- SciPy

---

# 🖥️ Streamlit Dashboard Features

The interactive dashboard provides:

✅ Clinical Feature Input  
✅ Embryo Quality Prediction  
✅ Prediction Probability Scores  
✅ AI-Powered Fertility Insights  
✅ Responsive Medical Dashboard UI  
✅ Real-time Predictions  
✅ User-friendly Interface  

---

# 🚀 How to Run the Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Embryo-Quality-Prediction.git
cd Embryo-Quality-Prediction
```

---

## 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Add Dataset

Download a similar fertility dataset from Kaggle and place it inside the `dataset/` folder.

Update dataset path in the code:

```python
file_path = "dataset/fertility_dataset.csv"
```

---

## 4️⃣ Run EDA & Preprocessing

```bash
python eda_preprocessing.py
```

---

## 5️⃣ Train Machine Learning Models

```bash
python model_training.py
```

---

## 6️⃣ Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 📊 Output Files Generated

The system generates:

- Cleaned Dataset CSV
- Trained ML Models (.pkl)
- Label Encoder
- Scaler File
- Accuracy Comparison Charts
- F1 Score Comparison Charts
- SHAP Explainability Graphs

---

# 📌 Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
EDA & Visualization
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
SHAP Explainability
   ↓
Model Saving
   ↓
Streamlit Deployment
```

---

# 📷 Dashboard Preview

Add dashboard screenshots here:

```bash
images/dashboard_preview.png
```

---

# 👨‍💻 Author

## K SAI PRAGNA

### Project:
Embryo Quality Prediction Using PGx - AI & ML

---

# 📜 License

This project is developed for educational and research purposes only.

---

# ⭐ Future Enhancements

- Deep Learning Models
- Real-time IVF Analytics
- Cloud Deployment
- Explainable Medical AI
- Multi-Dataset Fertility Prediction
- Advanced Clinical Decision Support Systems

---

# 🙌 Acknowledgements

- Scikit-learn
- XGBoost
- Streamlit
- SHAP Community
- Open Source Community