# === IMPORTS ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === LOAD DATA ===
df = pd.read_csv(r"C:\Users\PRAGNA\OneDrive\Desktop\EDA_FINAL_6_FEATURES.csv")

# === FEATURE SELECTION ===
features = ['FSH(mIU/mL)', 'LH(mIU/mL)', 'Age (yrs)', 'AMH(ng/mL)', 'BMI', 'AFC']

# === TARGET CREATION ===
def classify_afc(afc):
    if afc <= 8:
        return 'Poor'
    elif 9 <= afc <= 12:
        return 'Fair'
    else:
        return 'Best'

df['Embryo_Quality'] = df['AFC'].apply(classify_afc)

# === FEATURES & LABELS ===
X = df[features]
y = df['Embryo_Quality']

# === ENCODING ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === SCALING ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === MODELS ===
models = {
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = []
best_f1 = 0
best_model = None
best_model_name = ""

# === TRAINING ===
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({'Model': name, 'Accuracy': acc, 'F1_macro': f1})

    print(f"✅ {name} Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# === SAVE BEST MODEL ===
joblib.dump(best_model, r"C:\Users\PRAGNA\OneDrive\Desktop\XGBModel.pkl")
joblib.dump(le, r"C:\Users\PRAGNA\OneDrive\Desktop\EQP_label_encoder.pkl")
joblib.dump(scaler, r"C:\Users\PRAGNA\OneDrive\Desktop\Scaler_EQP.pkl")
print(f"\n✅ Best model '{best_model_name}' saved successfully!")

# === MODEL COMPARISON ===
results_df = pd.DataFrame(results)
results_df['Accuracy_Percent'] = results_df['Accuracy'] * 100
results_df['F1_macro_Percent'] = results_df['F1_macro'] * 100

print("\n📊 Model Comparison:")
print(results_df[['Model', 'Accuracy_Percent', 'F1_macro_Percent']])

# === ACCURACY PLOT  ===
acc_df = results_df.sort_values('Accuracy_Percent', ascending=False)

plt.figure(figsize=(14, 6))
sns.barplot(x='Accuracy_Percent', y='Model', data=acc_df, hue="Model", palette="rocket", legend=False)
plt.title("Model Accuracy Comparison (High to Low)", fontsize=16, fontweight='bold')
plt.xlabel("Accuracy_Percent", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.xlim(0, 100)  # make some space on right

plt.grid(axis='x', linestyle='--', alpha=0.5)
for i, (acc, model) in enumerate(zip(acc_df["Accuracy_Percent"], acc_df["Model"])):
    plt.text(acc + 1, i, f"{acc:.2f}%", va='center', fontsize=10)
plt.tight_layout()
plt.show()


# === F1 SCORE PLOT ===
f1_df = results_df.sort_values('F1_macro_Percent', ascending=False)

plt.figure(figsize=(14, 6))
sns.barplot(x='F1_macro_Percent', y='Model', data=f1_df, hue="Model", palette="rocket", legend=False)
plt.title("Model F1 Score Comparison (High to Low)", fontsize=16, fontweight='bold')
plt.xlabel("F1 Score (%)", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.xlim(0, 100)

plt.grid(axis='x', linestyle='--', alpha=0.5)
for i, (f1, model) in enumerate(zip(f1_df["F1_macro_Percent"], f1_df["Model"])):
    plt.text(f1 + 1, i, f"{f1:.2f}%", va='center', fontsize=10)
plt.tight_layout()
plt.show()

# === SHAP EXPLAINABILITY (only for XGBoost) ===
if best_model_name == "XGBoost":
    # Use original scaled features (X_train)
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, features=X.columns, plot_type="bar")
