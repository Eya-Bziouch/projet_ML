#STEP 1 : Loading the dataset + exploring 
#adding Bar chart 
import matplotlib
# Use a non-interactive backend so plots work even when Tk/Tcl is unavailable.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd #1

df = pd.read_csv('heart_disease.csv') 
print("Dataset loaded successfully! before cleaning and preparation.")
print("First 5 rows of the dataset: ")
print(df.head()) # show first 5 rows
print("**********************************")
print("Dataset information: ")
df.info() # show column types 
print("**********************************")
print("Dataset statistics: ")
print(df.describe()) # show statistics 
print("**********************************")
print("Missing values: ")
print(df.isnull().sum()) # check for missing values

#many people in each class
"""sns.countplot(x='num', data=df)
plt.title("Distribution of Heart Disease Classes")
plt.savefig('heart_disease_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()"""

#STEP 2 : Clean And Prepare the data 
#2
import numpy as np
# from sklearn.impute import SimpleImputer  # ❌ removed (not used)

# 1. Drop the 'id' column — it's just a row number, not useful
df = df.drop(columns=['id'])

# 2. Replace impossible 0s with NaN (chol and trestbps)
df['chol']     = df['chol'].replace(0, np.nan)
df['trestbps'] = df['trestbps'].replace(0, np.nan)

# 3. For columns with >50% missing (ca, thal) → DROP them entirely
#    OR keep and impute — but mention this choice in your report
df = df.drop(columns=['ca', 'thal'])   # safest option

# 4. Encode categorical text columns as numbers
df = pd.get_dummies(df, columns=['sex','cp','restecg','slope','fbs','exang','dataset'])

# 5. Fill remaining missing numeric values with the median
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("Data cleaning and preparation completed.")
print("First 5 rows of the cleaned dataset: ")
print(df.head())
df.info()
print(df.isnull().sum())
print("Shape:", df.shape)
print(df.describe())

"""sns.countplot(x='num', data=df)
plt.title("Target distribution after cleaning")
plt.savefig('target_distribution_after_cleaning.png', dpi=150, bbox_inches='tight')
plt.close()
"""

#STEP 3 : Algorithm #1: KNN (K-Nearest Neighbors)
#3knn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split X and y
X = df.drop('num', axis=1)
y = df['num']

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training KNN model...")

# Train model
knn = KNeighborsClassifier(n_neighbors=9, weights='distance')
knn.fit(X_train, y_train)

# Predict
y_pred_knn = knn.predict(X_test)

# Evaluate
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn, zero_division=0))
"""
cm = confusion_matrix(y_test, y_pred_knn)
print("\nConfusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('knn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved as knn_confusion_matrix.png")

"""

#STEP 4 : Algorithm #2: Decision Tree 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Training Decision Tree model...")
# Create model
dt = DecisionTreeClassifier(random_state=42)

# Train
dt.fit(X_train, y_train)

# Predict
y_pred_dt = dt.predict(X_test)

# Evaluate
print("Decision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

'''plt.figure(figsize=(20,12))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4'])
plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Decision tree visualization saved as decision_tree_visualization.png")'''

#STEP 5 : Algorihm #3 : Random Forest 
print("Training Random Forest model...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Evaluate
print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

importances = rf.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df)

# Plot
plt.figure(figsize=(10,5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance plot saved as random_forest_feature_importance.png")

#STEP 6 :  Ensemble Learning (AdaBoost)

print("Training AdaBoost model with Decision Tree base estimator...")
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("Training AdaBoost model...")
# -----------------------------
# 1. Train AdaBoost model
# -----------------------------
# FIX 🔴 Bug 3: removed algorithm parameter (not supported in this scikit-learn version)
ada = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

ada.fit(X_train, y_train)

# -----------------------------
# 2. Predictions
# -----------------------------
y_pred_ada = ada.predict(X_test)

# -----------------------------
# 3. Accuracy
# -----------------------------
acc = accuracy_score(y_test, y_pred_ada)
print("AdaBoost Accuracy:", acc)

# -----------------------------
# 4. Confusion Matrix Visualization
# -----------------------------
"""
cm = confusion_matrix(y_test, y_pred_ada)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("AdaBoost - Confusion Matrix")

# SAVE IMAGE
plt.savefig("adaboost_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("AdaBoost confusion matrix saved as adaboost_confusion_matrix.png")
"""

# STEP 7 :  RIPPER (Rule-Based)
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import wittgenstein as lw

print("Training RIPPER model (Binary Classification)...")
# RIPPER only supports binary classification
# Convert to binary: 0 = No Disease, 1 = Has Disease (any level 1-4)
y_train_binary = (y_train > 0).astype(int)
y_test_binary = (y_test > 0).astype(int)

# Convert to DataFrame (IMPORTANT for RIPPER)
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Train RIPPER model
ripper = lw.RIPPER()
ripper.fit(X_train_df, y=y_train_binary, pos_class=1)

# Predictions
y_pred_rip = ripper.predict(X_test_df)

# Accuracy (binary classification)
ripper_accuracy = accuracy_score(y_test_binary, y_pred_rip)
print(f"RIPPER Accuracy (Binary): {ripper_accuracy:.4f}")

# Classification Report
print("\nClassification Report (Binary):")
print(classification_report(y_test_binary, y_pred_rip, zero_division=0))
'''
# Confusion Matrix
cm_rip = confusion_matrix(y_test_binary, y_pred_rip)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rip, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Has Disease'], yticklabels=['No Disease', 'Has Disease'])
plt.title('RIPPER Confusion Matrix (Binary Classification)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('ripper_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("RIPPER confusion matrix saved as ripper_confusion_matrix.png")

# 4. Summary
# ----------------------------------
acc = accuracy_score(y_test, y_pred_rip)
print("RIPPER Accuracy:", acc)

# -----------------------------
# 5. Confusion Matrix (Visualization)
# -----------------------------
cm = confusion_matrix(y_test, y_pred_rip)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Greens")

plt.title("RIPPER - Confusion Matrix")

# SAVE IMAGE
plt.savefig("ripper_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("RIPPER confusion matrix saved as ripper_confusion_matrix.png")
'''

#STEP 8 :  Evaluate Every Model 
print("evaluation models ")
'''
labels = ['No disease', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model_name, y_true, y_pred, labels):
    print("\n==============================")
    print(f"MODEL: {model_name}")
    print("==============================\n")

    # -----------------------------
    # A) Classification Report
    # -----------------------------
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    # -----------------------------
    # B) Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')

    plt.title(f"{model_name} - Confusion Matrix")

    # Save figure
    filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

evaluate_model("KNN",           y_test,        y_pred_knn, labels)
evaluate_model("Decision Tree", y_test,        y_pred_dt,  labels)          # ✅ FIX Bug 1: was y_pred_tree (NameError)
evaluate_model("Random Forest", y_test,        y_pred_rf,  labels)
evaluate_model("AdaBoost",      y_test,        y_pred_ada, labels)
evaluate_model("RIPPER",        y_test_binary, y_pred_rip, ['No Disease', 'Has Disease'])  # ✅ FIX Bug 2: binary labels + binary y_test

'''

# multioutput 

print("Training MultiOutputClassifier with Random Forest base estimator...")

from sklearn.ensemble import RandomForestClassifier

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

print("Model training completed successfully!")

# =========================================================
# 📊 VISUALIZATION: Feature Importance (IMPORTANT PART)
# =========================================================

# Get feature importance
importances = model.feature_importances_

# If X_train is a DataFrame, keep column names
if isinstance(X_train, pd.DataFrame):
    feature_names = X_train.columns
else:
    feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

# Create DataFrame for better visualization
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort values
feat_df = feat_df.sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(feat_df["Feature"], feat_df["Importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

# Save image
plt.savefig("feature_importance.png")

# Close plot
plt.close()