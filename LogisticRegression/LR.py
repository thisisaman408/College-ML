# -*- coding: utf-8 -*-
"""
Logistic Regression for Mobile Price Range Prediction

Alright, time to tackle Bob's mobile price range prediction using Logistic Regression.
This is interesting because the target 'price_range' has 4 categories (0, 1, 2, 3),
making it a multi-class classification problem. Scikit-learn's LogisticRegression
can handle this using One-vs-Rest (OvR) or Multinomial (Softmax) approaches.

My plan is:
1. Load the specific 'train.csv' and 'test.csv' files mentioned.
2. Preprocess: Mainly focus on scaling the features, as Logistic Regression,
   especially with regularization, benefits from it.
3. Baseline Model: Train a default Logistic Regression model (likely L2 regularized).
4. Probability Analysis: Look at the predicted probabilities for each class and
   understand how the final prediction is made (no single threshold like binary).
5. Regularization Deep Dive:
    - Explore L2 (Ridge) regularization by varying the 'C' parameter.
    - Explore L1 (Lasso) regularization by varying 'C'.
    - Compare performance and visualize how coefficients change. See if L1 performs feature selection.
6. Evaluation: Use accuracy, confusion matrix, classification report, and log loss.
7. Save plots and results clearly in 'LogisticRegression_Outputs'.
8. Maintain the expressive, first-person commentary style.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Scikit-learn imports
from sklearn.model_selection import train_test_split # Although using separate files, might need if combining
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

# Ignore convergence warnings for now, might need to increase max_iter later
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- 1. Setup and Data Loading ---

print("--- Setting up Environment and Loading Data ---")

# Directory for saving plots
output_dir = 'LogisticRegression_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: '{output_dir}' to save my plots.")
else:
    print(f"Directory '{output_dir}' already exists. Plots will be saved there.")

# Load the specific train and test datasets
try:
    train_file = 'train.csv' # Assumes train.csv is in the same directory
    test_file = 'test.csv'   # Assumes test.csv is in the same directory
    df_train = pd.read_csv(train_file)
    # The test file provided in some Kaggle competitions might not have the target column.
    # Let's check. If 'price_range' is missing in test.csv, we can only use train.csv
    # and split it ourselves. Let's assume test.csv *also* has the price_range for evaluation for now.
    # If not, I'll need to split df_train instead.
    try:
        df_test = pd.read_csv(test_file)
        print(f"Successfully loaded training data from '{train_file}' ({df_train.shape}).")
        print(f"Successfully loaded testing data from '{test_file}' ({df_test.shape}).")
        # Verify 'price_range' is in both
        if 'price_range' not in df_test.columns:
             print("Warning: 'price_range' column not found in test.csv. Will split train.csv instead.")
             # Combine, then split - or just split train
             print("Splitting train.csv into training and validation sets (80/20).")
             df_full = df_train.copy()
             df_train, df_test = train_test_split(df_full, test_size=0.20, random_state=42, stratify=df_full['price_range'])
             print(f"New training set size: {df_train.shape}")
             print(f"New validation set size: {df_test.shape}")

    except FileNotFoundError:
         print(f"ERROR: Could not find the file '{test_file}'. Will split train.csv.")
         df_full = df_train.copy()
         df_train, df_test = train_test_split(df_full, test_size=0.20, random_state=42, stratify=df_full['price_range'])
         print(f"New training set size: {df_train.shape}")
         print(f"New validation set size: {df_test.shape}")


except FileNotFoundError:
    print(f"ERROR: Could not find the file '{train_file}'. Cannot proceed.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- 2. Data Preprocessing ---

print("\n--- Starting Data Preprocessing ---")

# Separate features (X) and target variable (y)
X_train = df_train.drop('price_range', axis=1)
y_train = df_train['price_range']

X_test = df_test.drop('price_range', axis=1)
y_test = df_test['price_range']

print(f"Training features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Testing target shape: {y_test.shape}")

# Check for missing values (already seemed okay from previous info, but good to double-check)
print("\nChecking for missing values in training data:")
print(X_train.isnull().sum().sum()) # Total missing values
print("Checking for missing values in testing data:")
print(X_test.isnull().sum().sum()) # Total missing values
# Assuming 0 based on previous info.

# Feature Scaling: Use StandardScaler
# Logistic Regression converges faster and regularization works better with scaled features.
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform both training and test data using the SAME scaler
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete.")
print(f"Mean of 'ram' in scaled training data: {X_train_scaled[:, X_train.columns.get_loc('ram')].mean():.4f} (should be ~0)")
print(f"Std Dev of 'ram' in scaled training data: {X_train_scaled[:, X_train.columns.get_loc('ram')].std():.4f} (should be ~1)")


# --- 3. Baseline Logistic Regression Model ---

print("\n--- Training Baseline Logistic Regression Model ---")
# Using default parameters: C=1.0, penalty='l2', multi_class='auto' (likely chooses 'ovr' or 'multinomial')
# Using 'lbfgs' solver which is good for multinomial and supports L2.
# Increased max_iter just in case.
log_reg_base = LogisticRegression(
    C=1.0, penalty='l2', multi_class='auto', solver='lbfgs', max_iter=1000, random_state=42
)

print(f"Training baseline model with parameters: C={log_reg_base.C}, penalty={log_reg_base.penalty}, solver={log_reg_base.solver}, multi_class={log_reg_base.multi_class}") 

log_reg_base.fit(X_train_scaled, y_train)
print("Baseline model training complete.")

# Evaluate the baseline model
print("\nEvaluating baseline model on the test set...")
y_pred_base = log_reg_base.predict(X_test_scaled)
y_prob_base = log_reg_base.predict_proba(X_test_scaled)

accuracy_base = accuracy_score(y_test, y_pred_base)
logloss_base = log_loss(y_test, y_prob_base) # Log loss uses probabilities
cm_base = confusion_matrix(y_test, y_pred_base)
report_base = classification_report(y_test, y_pred_base)

print(f"\nAccuracy (Baseline): {accuracy_base:.4f}")
print(f"Log Loss (Baseline): {logloss_base:.4f}") # Lower is better
print("\nClassification Report (Baseline):\n", report_base)
print("\nConfusion Matrix (Baseline):")
print(cm_base)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=log_reg_base.classes_, yticklabels=log_reg_base.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Baseline Logistic Regression\nAccuracy: {accuracy_base:.3f}")
plot_path = os.path.join(output_dir, '01_confusion_matrix_baseline.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved baseline confusion matrix plot to: {plot_path}")

# --- 4. Probability Analysis ---

print("\n--- Analyzing Predicted Probabilities (Baseline Model) ---")
# Let's look at the probabilities for the first few test samples.
# The shape should be (n_samples, n_classes)
print(f"Shape of predicted probabilities: {y_prob_base.shape}")
print("\nPredicted Probabilities for first 5 test samples:")
# Nicely format the probabilities
prob_df_head = pd.DataFrame(y_prob_base[:5], columns=[f'P(Class={c})' for c in log_reg_base.classes_])
prob_df_head['Predicted Label'] = y_pred_base[:5]
prob_df_head['Actual Label'] = y_test.iloc[:5].values
print(prob_df_head.round(4))

print("\nExplanation:")
print("For each sample (row), the model calculates the probability of it belonging to each class (0, 1, 2, 3).")
print("The final 'Predicted Label' is the class with the highest probability (argmax).")
print("In multi-class logistic regression, there isn't a single 'threshold' like in binary cases.")
print("Instead, we compare the probabilities between classes.")
print("Adjusting decision logic based on these probabilities is possible but depends on specific needs (e.g., cost-sensitive learning).")


# --- 5. Impact of L2 (Ridge) Regularization ---

print("\n--- Exploring L2 (Ridge) Regularization ---")
# L2 regularization adds a penalty proportional to the square of the coefficients.
# It shrinks coefficients towards zero, helping to prevent overfitting.
# 'C' is the *inverse* of regularization strength: Smaller C = Stronger regularization.

C_values_l2 = [0.001, 0.01, 0.1, 1, 10, 100] # Test a range of C values
l2_results = {}
l2_coefficients = {}

print(f"Testing L2 regularization with C values: {C_values_l2}")

for C in C_values_l2:
    print(f"\nTraining model with L2, C={C}...")
    # Using 'lbfgs' solver which works well with L2
    log_reg_l2 = LogisticRegression(C=C, penalty='l2', multi_class='auto', solver='lbfgs', max_iter=1000, random_state=42)
    log_reg_l2.fit(X_train_scaled, y_train)

    y_pred_l2 = log_reg_l2.predict(X_test_scaled)
    y_prob_l2 = log_reg_l2.predict_proba(X_test_scaled)

    accuracy_l2 = accuracy_score(y_test, y_pred_l2)
    logloss_l2 = log_loss(y_test, y_prob_l2)
    print(f"  Accuracy: {accuracy_l2:.4f}, Log Loss: {logloss_l2:.4f}")

    l2_results[C] = {'accuracy': accuracy_l2, 'logloss': logloss_l2}
    l2_coefficients[C] = log_reg_l2.coef_.copy() # Store coefficients (shape: n_classes x n_features)

# Compare performance across different C values for L2
print("\nL2 Regularization Performance Summary:")
print("  C    | Accuracy | Log Loss")
print("-----------------------------")
for C, metrics in l2_results.items():
    print(f"{C:<6} | {metrics['accuracy']:.4f}   | {metrics['logloss']:.4f}")

# Visualize coefficient magnitudes for L2
# Let's plot the L2 norm (magnitude) of coefficients for each class vs. C
plt.figure(figsize=(12, 8))
coef_norms_l2 = {c: [] for c in log_reg_base.classes_} # Coeffs per class
for C_val, coefs in l2_coefficients.items():
    for i, class_label in enumerate(log_reg_base.classes_):
        norm = np.linalg.norm(coefs[i, :]) # L2 norm of coefficient vector for class i
        coef_norms_l2[class_label].append(norm)

for class_label, norms in coef_norms_l2.items():
     # Use log scale for C axis for better visualization
    plt.plot(np.log10(C_values_l2), norms, marker='o', label=f'Class {class_label}')

plt.xlabel("Log10(C) (Inverse Regularization Strength)")
plt.ylabel("L2 Norm (Magnitude) of Coefficients")
plt.title("Impact of L2 Regularization (C) on Coefficient Magnitudes")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '02_l2_coeffs_magnitude_vs_C.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved L2 coefficient magnitude plot to: {plot_path}")
print("Observation: As C decreases (stronger regularization), the magnitude of coefficients shrinks.")


# --- 6. Impact of L1 (Lasso) Regularization ---

print("\n--- Exploring L1 (Lasso) Regularization ---")
# L1 regularization adds a penalty proportional to the absolute value of coefficients.
# It can force some coefficients to become exactly zero, performing feature selection.
# We need a solver that supports L1, like 'liblinear' (only OvR) or 'saga'.
# Let's use 'saga' as it's versatile. It might need more iterations.

C_values_l1 = [0.001, 0.01, 0.1, 1, 10, 100] # Test similar C values
l1_results = {}
l1_coefficients = {}
l1_zero_counts = {}

print(f"Testing L1 regularization with C values: {C_values_l1} (using 'saga' solver)")

for C in C_values_l1:
    print(f"\nTraining model with L1, C={C}...")
    # multi_class='auto' with saga/L1 might default to OvR, let's try that explicitly if needed
    log_reg_l1 = LogisticRegression(C=C, penalty='l1', multi_class='auto', solver='saga', max_iter=3000, random_state=42) # Increased max_iter for saga
    log_reg_l1.fit(X_train_scaled, y_train)

    y_pred_l1 = log_reg_l1.predict(X_test_scaled)
    y_prob_l1 = log_reg_l1.predict_proba(X_test_scaled)

    accuracy_l1 = accuracy_score(y_test, y_pred_l1)
    logloss_l1 = log_loss(y_test, y_prob_l1)
    num_zeros = np.sum(np.abs(log_reg_l1.coef_) < 1e-6) # Count near-zero coefficients across all classes
    print(f"  Accuracy: {accuracy_l1:.4f}, Log Loss: {logloss_l1:.4f}, Zero Coeffs: {num_zeros}")

    l1_results[C] = {'accuracy': accuracy_l1, 'logloss': logloss_l1}
    l1_coefficients[C] = log_reg_l1.coef_.copy()
    l1_zero_counts[C] = num_zeros


# Compare performance across different C values for L1
print("\nL1 Regularization Performance Summary:")
print("  C    | Accuracy | Log Loss | Zero Coeffs")
print("--------------------------------------------")
for C, metrics in l1_results.items():
    zeros = l1_zero_counts[C]
    print(f"{C:<6} | {metrics['accuracy']:.4f}   | {metrics['logloss']:.4f} | {zeros}")

# Visualize the number of zero coefficients for L1
plt.figure(figsize=(10, 6))
n_features_total = l1_coefficients[C_values_l1[0]].size # Total coefficients = n_classes * n_features
non_zero_counts = [n_features_total - l1_zero_counts[C] for C in C_values_l1]

plt.plot(np.log10(C_values_l1), non_zero_counts, marker='o', color='red')
plt.xlabel("Log10(C) (Inverse Regularization Strength)")
plt.ylabel("Number of Non-Zero Coefficients")
plt.title("Impact of L1 Regularization (C) on Feature Selection")
plt.grid(True, linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '03_l1_non_zero_coeffs_vs_C.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved L1 non-zero coefficients plot to: {plot_path}")
print("Observation: As C decreases (stronger L1 regularization), more coefficients become exactly zero.")
print("This demonstrates L1's feature selection capability.")

# Visualize coefficients for a specific feature (e.g., 'ram') across classes for L1
ram_index = X_train.columns.get_loc('ram')
plt.figure(figsize=(12, 8))

for i, class_label in enumerate(log_reg_base.classes_):
    ram_coeffs = [l1_coefficients[C][i, ram_index] for C in C_values_l1]
    plt.plot(np.log10(C_values_l1), ram_coeffs, marker='o', label=f'Class {class_label} Coeff for RAM')

plt.xlabel("Log10(C) (Inverse Regularization Strength)")
plt.ylabel("Coefficient Value for RAM")
plt.title("Impact of L1 Regularization (C) on 'RAM' Coefficient")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '04_l1_ram_coeff_vs_C.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved L1 RAM coefficient plot to: {plot_path}")


# --- 7. Cross-Validation for Hyperparameter Tuning (Optional) ---

print("\n--- Tuning Regularization Strength (C) using Cross-Validation (L2 Example) ---")
# LogisticRegressionCV performs CV internally to find the best C from a list.
# Let's try it for L2 penalty. It uses 'lbfgs' by default.
# Cs=10 means it tests 10 values in a logarithmic scale between 1e-4 and 1e4 by default.
# cv=5 means 5-fold cross-validation.
print("Running LogisticRegressionCV with L2 penalty...")
log_reg_cv = LogisticRegressionCV(
    Cs=10, cv=5, penalty='l2', multi_class='auto', solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1 # Use all CPU cores
)

log_reg_cv.fit(X_train_scaled, y_train)

best_C_l2 = log_reg_cv.C_[0] # C_ is an array, one value per class in OvR, usually same for multinomial
print(f"\nBest C found by LogisticRegressionCV (L2): {best_C_l2:.4f}")

# Evaluate the CV-tuned model
print("\nEvaluating the CV-tuned L2 model on the test set...")
y_pred_cv = log_reg_cv.predict(X_test_scaled)
y_prob_cv = log_reg_cv.predict_proba(X_test_scaled)

accuracy_cv = accuracy_score(y_test, y_pred_cv)
logloss_cv = log_loss(y_test, y_prob_cv)
cm_cv = confusion_matrix(y_test, y_pred_cv)
report_cv = classification_report(y_test, y_pred_cv)

print(f"\nAccuracy (CV-Tuned L2): {accuracy_cv:.4f}")
print(f"Log Loss (CV-Tuned L2): {logloss_cv:.4f}")
print("\nClassification Report (CV-Tuned L2):\n", report_cv)
print("\nConfusion Matrix (CV-Tuned L2):")
print(cm_cv)

# Visualize Confusion Matrix for CV-tuned model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Purples', xticklabels=log_reg_cv.classes_, yticklabels=log_reg_cv.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - CV-Tuned L2 Logistic Regression (C={best_C_l2:.2f})\nAccuracy: {accuracy_cv:.3f}")
plot_path = os.path.join(output_dir, '05_confusion_matrix_CV_tuned_L2.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved CV-tuned L2 confusion matrix plot to: {plot_path}")


# --- 8. Final Summary ---

print("\n--- Final Model Performance Summary ---")
print("--------------------------------------------------------------------")
print(" Model                     | Penalty |    C    | Accuracy | Log Loss")
print("--------------------------------------------------------------------")
print(f" Baseline (Default L2)     | L2      | {log_reg_base.C:<7.3f} | {accuracy_base:.4f}   | {logloss_base:.4f}")
# Find best performing L2 from manual tests
best_l2_C = max(l2_results, key=lambda k: l2_results[k]['accuracy'])
print(f" Manual Best L2            | L2      | {best_l2_C:<7.3f} | {l2_results[best_l2_C]['accuracy']:.4f}   | {l2_results[best_l2_C]['logloss']:.4f}")
# Find best performing L1 from manual tests
best_l1_C = max(l1_results, key=lambda k: l1_results[k]['accuracy'])
print(f" Manual Best L1            | L1      | {best_l1_C:<7.3f} | {l1_results[best_l1_C]['accuracy']:.4f}   | {l1_results[best_l1_C]['logloss']:.4f}")
print(f" CV Tuned L2               | L2      | {best_C_l2:<7.3f} | {accuracy_cv:.4f}   | {logloss_cv:.4f}")
print("--------------------------------------------------------------------")


print("\nKey takeaways from Logistic Regression analysis:")
print(" - Logistic Regression handles multi-class problems effectively (using OvR or Multinomial).")
print(" - Feature scaling is important for performance and convergence.")
print(" - Predicted probabilities show the model's confidence for each class.")
print(" - Regularization (L1/L2) is controlled by parameter C (inverse strength).")
print(" - L2 (Ridge) shrinks coefficients, reducing model complexity.")
print(" - L1 (Lasso) shrinks coefficients and performs feature selection (setting some to zero).")
print(" - Performance (accuracy, log loss) varies with the choice of penalty (L1/L2) and strength (C).")
print(" - Cross-validation (LogisticRegressionCV) can automate finding a good regularization strength.")

print("\n\n--- Logistic Regression Analysis Complete ---")
print(f"All plots saved in the '{output_dir}' folder.")