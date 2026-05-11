# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# STEP 1: Load dataset
file_path = r"DATASET_PATH.csv"  # Update with your actual dataset path
df = pd.read_csv(file_path)

# STEP 2: Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# STEP 3: Create AFC column (sum of Follicle No. (L) and Follicle No. (R))
if 'Follicle No. (L)' in df.columns and 'Follicle No. (R)' in df.columns:
    df['AFC'] = pd.to_numeric(df['Follicle No. (L)'], errors='coerce') + pd.to_numeric(df['Follicle No. (R)'], errors='coerce')

# STEP 4: Clean AMH column (convert to numeric, handle commas and spaces)
if 'AMH(ng/mL)' in df.columns:
    df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'].astype(str).str.replace(",", "").str.strip(), errors='coerce')

# STEP 5: Drop unwanted columns (unnamed columns etc.)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# STEP 6: Handle missing values (only for AMH column, since Cycle column is removed)
if 'AMH(ng/mL)' in df.columns:
    df['AMH(ng/mL)'] = df['AMH(ng/mL)'].fillna(df['AMH(ng/mL)'].median())

# STEP 7: Convert all columns possible to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# STEP 8: Select final 6 features (Cycle column removed)
features = [
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "Age (yrs)",
    "AMH(ng/mL)",
    "BMI",
    "AFC"
]

# STEP 9: Keep only selected features for further EDA
eda_df = df[features]

# STEP 10: Basic info
print("Dataset shape:", eda_df.shape)
print("\nData Types:\n", eda_df.dtypes)
print("\nFirst 5 rows:\n", eda_df.head())
print("\nSummary statistics:\n", eda_df.describe())

# STEP 11: Missing values check after imputation
print("\nMissing values after imputation:\n", eda_df.isnull().sum())

# STEP 12: Distribution plots for all features
eda_df[features].hist(bins=30, figsize=(15, 12), color='skyblue', edgecolor='black')
plt.suptitle("Distributions of Features", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# STEP 13: Boxplots for outlier detection
plt.figure(figsize=(20, 15))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=eda_df[col], color='skyblue')
    plt.title(col)
    plt.tight_layout()
plt.suptitle("Boxplots for All Features (Better View)", fontsize=20)
plt.tight_layout()
plt.show()

# STEP 14: Pairplot to check pairwise relationships
sns.pairplot(eda_df, diag_kind='kde', corner=True)
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

# STEP 15: Correlation matrix
correlation = eda_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# STEP 16: Skewness check
skewness = eda_df.skew().sort_values(ascending=False)
print("\nSkewness of Features:\n", skewness)

# STEP 17: Q-Q Plots for normality check
plt.figure(figsize=(20, 15))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)
    stats.probplot(eda_df[col], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {col}")
plt.tight_layout()
plt.show()

# STEP 18: Save final cleaned dataset for model building
eda_df.to_csv(r"C:\Users\PRAGNA\OneDrive\Desktop\EDA_FINAL_6_FEATURES.csv", index=False)
print("✅ Final cleaned dataset with 6 features saved after EDA.")
