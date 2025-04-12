# -*- coding: utf-8 -*-
"""
Demonstration of Linear Regression with Regularization and Cross-Validation

Okay, so my goal here is to explore Linear Regression using scikit-learn.
I'll start with the basic version, then add regularization techniques like
Ridge, Lasso, and ElasticNet to see how they affect the model, especially the coefficients.
Finally, I'll use cross-validation to get a more robust measure of performance
and maybe even tune hyperparameters like the regularization strength (alpha).
I need to make sure I compare everything clearly and save visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression # Using this for controlled data generation

# --- 1. Setup: Environment and Data ---

print("--- Setting up the Environment and Generating Data ---")

# First, I need a place to save my plots. Let's create a folder.
output_dir = 'LinearRegression_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: '{output_dir}' to save my plots.")
else:
    print(f"Directory '{output_dir}' already exists. Plots will be saved there.")

# Now, let's generate some synthetic data. This way, I know the underlying
# relationship and can see how well the models recover it.
# I'll create data with a few informative features and maybe some noise
# or less relevant features to see how regularization handles them.
n_samples = 10000
n_features = 10 # Let's have 10 features
n_informative = 5 # Only 5 of them will be truly useful
noise_level = 15  # Adding some noise to make it realistic
random_seed = 42 # For reproducibility, so I get the same data every time

print(f"\nGenerating synthetic dataset with:")
print(f" - {n_samples} samples")
print(f" - {n_features} total features")
print(f" - {n_informative} informative features")
print(f" - Noise level: {noise_level}")
print(f" - Random seed: {random_seed}")

X, y, true_coefficients = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    noise=noise_level,
    coef=True, # Ask the function to return the true coefficients it used
    random_state=random_seed
)

# It's good practice to split the data into training and testing sets.
# This way, I train the model on one part and test its performance on data
# it hasn't seen before.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed
)

print(f"\nSplit the data:")
print(f" - Training set size: {X_train.shape[0]} samples")
print(f" - Test set size: {X_test.shape[0]} samples")
print(f" - Number of features: {X_train.shape[1]}")

# Let's quickly visualize the true coefficients used to generate the data.
# This will be my ground truth for comparison later.
plt.figure(figsize=(10, 5))
plt.bar(range(n_features), true_coefficients)
plt.xlabel("Feature Index")
plt.ylabel("True Coefficient Value")
plt.title("True Coefficients Used to Generate Data")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '00_true_coefficients.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close() # Close the plot to free memory
print(f"\nSaved plot of true coefficients to: {plot_path}")
print("Notice how some coefficients are zero - these correspond to the non-informative features.")

# --- 2. Standard Linear Regression (Baseline) ---

print("\n--- Part 2: Standard Linear Regression (No Regularization, No Scaling yet) ---")

# Okay, let's start with the most basic model: LinearRegression.
# It tries to find the line (or hyperplane) that best fits the training data
# by minimizing the sum of squared differences between predictions and actual values.
lr_model = LinearRegression()

# Train the model using the training data.
print("Training the standard Linear Regression model...")
lr_model.fit(X_train, y_train)
print("Training complete.")

# Let's see the coefficients (weights) the model learned.
# These should ideally be close to the 'true_coefficients' I plotted earlier,
# but noise and the limited training data will cause differences.
lr_coefficients = lr_model.coef_
lr_intercept = lr_model.intercept_

print(f"\nLearned Intercept (bias): {lr_intercept:.4f}")
print(f"Learned Coefficients (weights):")
for i, coef in enumerate(lr_coefficients):
    print(f"  Feature {i}: {coef:.4f}")

# Now, let's compare these learned coefficients with the true ones visually.
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(n_features)
plt.bar(index - bar_width/2, true_coefficients, bar_width, label='True Coefficients', color='skyblue')
plt.bar(index + bar_width/2, lr_coefficients, bar_width, label='Learned Coefficients (LR)', color='salmon')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Comparison: True Coefficients vs. Learned Coefficients (Standard LR)")
plt.xticks(index)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '01_coefficients_LR_vs_True.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved plot comparing LR coefficients to true coefficients: {plot_path}")
print("I can see the model tried to capture the important features, but the coefficients might be noisy.")

# Let's evaluate the model on the *test* set. How well does it generalize?
# I'll use Mean Squared Error (MSE) and R-squared (R2 score).
# MSE: Lower is better (average squared difference).
# R2: Closer to 1 is better (proportion of variance explained).
print("\nEvaluating the standard Linear Regression model on the test set...")
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"  Mean Squared Error (MSE): {mse_lr:.4f}")
print(f"  R-squared (R2 Score): {r2_lr:.4f}")

# I should also visualize the predictions against the actual values.
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_lr, alpha=0.7, edgecolors='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2, label='Perfect Prediction Line (y=x)')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred_lr)")
plt.title(f"Standard Linear Regression: Actual vs. Predicted\nMSE: {mse_lr:.2f}, R2: {r2_lr:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plot_path = os.path.join(output_dir, '02_predictions_LR.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved scatter plot of predictions vs actual values: {plot_path}")
print("This plot gives me a visual sense of the prediction accuracy.")

# --- 3. Feature Scaling ---

print("\n--- Part 3: Introducing Feature Scaling ---")

# Regularized models (Ridge, Lasso, ElasticNet) are sensitive to the scale of features.
# If features have very different ranges, the regularization penalty might unfairly
# penalize features with larger values. So, it's standard practice to scale the data first.
# I'll use StandardScaler, which transforms data to have zero mean and unit variance.

print("Initializing StandardScaler...")
scaler = StandardScaler()

# IMPORTANT: I should fit the scaler ONLY on the training data.
# Then, I transform both the training and test data using that *same* fitted scaler.
# This prevents 'data leakage' from the test set into the training process.
print("Fitting StandardScaler on the TRAINING data...")
scaler.fit(X_train)
print("Transforming (scaling) both TRAINING and TEST data...")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete.")
print(f"Example: Mean of first feature in scaled training data: {X_train_scaled[:, 0].mean():.4f} (should be close to 0)")
print(f"Example: Std Dev of first feature in scaled training data: {X_train_scaled[:, 0].std():.4f} (should be close to 1)")

# --- 4. Regularized Linear Regression ---

print("\n--- Part 4: Regularized Linear Regression (using SCALED data) ---")

# Now I'll apply Ridge, Lasso, and ElasticNet. These add a penalty term to the
# linear regression cost function to shrink the coefficients. This helps prevent
# overfitting and can make the model more robust, especially if features are correlated.
# I'll use the scaled data from now on for these models.

# --- 4a. Ridge Regression (L2 Regularization) ---
# Ridge adds a penalty proportional to the *square* of the coefficients (L2 norm).
# It shrinks coefficients towards zero but rarely makes them exactly zero.
# Alpha controls the strength of regularization (higher alpha = stronger penalty).
print("\n--- 4a. Ridge Regression (L2) ---")
ridge_alpha = 1.0 # A common default starting value
print(f"Initializing Ridge model with alpha = {ridge_alpha}...")
ridge_model = Ridge(alpha=ridge_alpha, random_state=random_seed)

print("Training Ridge model on SCALED training data...")
ridge_model.fit(X_train_scaled, y_train)
print("Training complete.")

# Let's look at the coefficients. I expect them to be smaller (shrunk) compared to standard LR.
ridge_coefficients = ridge_model.coef_
ridge_intercept = ridge_model.intercept_
print(f"\nRidge Learned Intercept: {ridge_intercept:.4f}")
print(f"Ridge Learned Coefficients (alpha={ridge_alpha}):")
for i, coef in enumerate(ridge_coefficients):
    print(f"  Feature {i}: {coef:.4f}")

# Compare Ridge coefficients to the original LR coefficients (learned on unscaled data)
# and the true coefficients. It's a bit complex because LR was on unscaled data,
# but let's primarily compare Ridge (on scaled) vs True.
plt.figure(figsize=(12, 7))
bar_width = 0.25
index = np.arange(n_features)
plt.bar(index - bar_width, true_coefficients, bar_width, label='True Coefficients', color='skyblue')
# Note: Comparing scaled coefficients directly to unscaled LR coefs isn't perfect, but shows shrinkage trend
# plt.bar(index, lr_coefficients, bar_width, label='LR Coefs (Unscaled Data)', color='lightcoral', alpha=0.7)
plt.bar(index + bar_width, ridge_coefficients, bar_width, label=f'Ridge Coefs (Scaled Data, alpha={ridge_alpha})', color='salmon')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title(f"Coefficient Comparison: True vs. Ridge (alpha={ridge_alpha})")
plt.xticks(index)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '03_coefficients_Ridge_vs_True.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved plot comparing Ridge coefficients: {plot_path}")
print("Observe how Ridge coefficients are generally smaller than the 'true' ones, especially for less important features.")
print("Unlike Lasso (next), they usually don't become exactly zero easily.")

# Evaluate Ridge on the SCALED test set.
print("\nEvaluating Ridge model on the SCALED test set...")
y_pred_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"  Ridge Mean Squared Error (MSE): {mse_ridge:.4f}")
print(f"  Ridge R-squared (R2 Score): {r2_ridge:.4f}")
print(f"  Comparison with Standard LR: MSE changed from {mse_lr:.4f} to {mse_ridge:.4f}")
print(f"                             R2 changed from {r2_lr:.4f} to {r2_ridge:.4f}")
# Sometimes regularization improves test performance (reduces overfitting), sometimes it might slightly decrease it if underfitting.

# Visualize Ridge predictions
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_ridge, alpha=0.7, edgecolors='k', label='Ridge Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred_ridge)")
plt.title(f"Ridge Regression (alpha={ridge_alpha}): Actual vs. Predicted\nMSE: {mse_ridge:.2f}, R2: {r2_ridge:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plot_path = os.path.join(output_dir, '04_predictions_Ridge.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved scatter plot of Ridge predictions: {plot_path}")


# --- 4b. Lasso Regression (L1 Regularization) ---
# Lasso adds a penalty proportional to the *absolute value* of the coefficients (L1 norm).
# A key difference from Ridge is that Lasso can force some coefficients to be exactly zero.
# This makes Lasso useful for feature selection.
# Alpha controls the strength (higher alpha = stronger penalty = more zeros).
print("\n--- 4b. Lasso Regression (L1) ---")
lasso_alpha = 1.0 # Let's start with alpha=1.0 again. Note: Lasso alpha scale can be different from Ridge.
print(f"Initializing Lasso model with alpha = {lasso_alpha}...")
lasso_model = Lasso(alpha=lasso_alpha, random_state=random_seed, max_iter=10000) # Increase max_iter if it doesn't converge

print("Training Lasso model on SCALED training data...")
lasso_model.fit(X_train_scaled, y_train)
print("Training complete.")

# Check the Lasso coefficients. I expect some might be zero now.
lasso_coefficients = lasso_model.coef_
lasso_intercept = lasso_model.intercept_
print(f"\nLasso Learned Intercept: {lasso_intercept:.4f}")
print(f"Lasso Learned Coefficients (alpha={lasso_alpha}):")
num_zero_coefs = np.sum(np.abs(lasso_coefficients) < 1e-6) # Count near-zero coefficients
for i, coef in enumerate(lasso_coefficients):
    print(f"  Feature {i}: {coef:.4f} {'(ZEROED OUT!)' if np.abs(coef) < 1e-6 else ''}")
print(f"\nLasso forced {num_zero_coefs} out of {n_features} coefficients to zero (or very close).")

# Compare Lasso coefficients to True and Ridge.
plt.figure(figsize=(12, 7))
bar_width = 0.25
index = np.arange(n_features)
plt.bar(index - bar_width, true_coefficients, bar_width, label='True Coefficients', color='skyblue')
plt.bar(index, ridge_coefficients, bar_width, label=f'Ridge Coefs (alpha={ridge_alpha})', color='lightcoral', alpha=0.7)
plt.bar(index + bar_width, lasso_coefficients, bar_width, label=f'Lasso Coefs (alpha={lasso_alpha})', color='salmon')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title(f"Coefficient Comparison: True vs. Ridge vs. Lasso (alpha={lasso_alpha})")
plt.xticks(index)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '05_coefficients_Lasso_vs_Others.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved plot comparing Lasso coefficients: {plot_path}")
print("Notice how Lasso aggressively sets some coefficients to zero, effectively performing feature selection.")

# Evaluate Lasso on the SCALED test set.
print("\nEvaluating Lasso model on the SCALED test set...")
y_pred_lasso = lasso_model.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"  Lasso Mean Squared Error (MSE): {mse_lasso:.4f}")
print(f"  Lasso R-squared (R2 Score): {r2_lasso:.4f}")
print(f"  Comparison with Ridge: MSE changed from {mse_ridge:.4f} to {mse_lasso:.4f}")
print(f"                       R2 changed from {r2_ridge:.4f} to {r2_lasso:.4f}")

# Visualize Lasso predictions
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_lasso, alpha=0.7, edgecolors='k', label='Lasso Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred_lasso)")
plt.title(f"Lasso Regression (alpha={lasso_alpha}): Actual vs. Predicted\nMSE: {mse_lasso:.2f}, R2: {r2_lasso:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plot_path = os.path.join(output_dir, '06_predictions_Lasso.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved scatter plot of Lasso predictions: {plot_path}")

# --- 4c. Elastic Net Regression ---
# Elastic Net combines both L1 (Lasso) and L2 (Ridge) penalties.
# It has two parameters:
#   - alpha: Controls the overall strength of regularization.
#   - l1_ratio: Controls the mix between L1 and L2 (0=Ridge, 1=Lasso, 0<l1_ratio<1 is mix).
# It can be useful when there are multiple correlated features; Lasso might pick one arbitrarily,
# while Elastic Net might shrink/select the group together.
print("\n--- 4c. Elastic Net Regression (L1 + L2) ---")
enet_alpha = 1.0 # Overall strength
enet_l1_ratio = 0.5 # Mix: 50% L1, 50% L2
print(f"Initializing ElasticNet model with alpha = {enet_alpha} and l1_ratio = {enet_l1_ratio}...")
enet_model = ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1_ratio, random_state=random_seed, max_iter=10000)

print("Training ElasticNet model on SCALED training data...")
enet_model.fit(X_train_scaled, y_train)
print("Training complete.")

# Check Elastic Net coefficients. I expect a mix of shrinkage and zeroing-out.
enet_coefficients = enet_model.coef_
enet_intercept = enet_model.intercept_
print(f"\nElasticNet Learned Intercept: {enet_intercept:.4f}")
print(f"ElasticNet Learned Coefficients (alpha={enet_alpha}, l1_ratio={enet_l1_ratio}):")
num_zero_coefs_enet = np.sum(np.abs(enet_coefficients) < 1e-6)
for i, coef in enumerate(enet_coefficients):
    print(f"  Feature {i}: {coef:.4f} {'(ZEROED OUT!)' if np.abs(coef) < 1e-6 else ''}")
print(f"\nElasticNet forced {num_zero_coefs_enet} coefficients to zero.")

# Compare Elastic Net coefficients to the others.
plt.figure(figsize=(14, 7))
bar_width = 0.2
index = np.arange(n_features)
plt.bar(index - 1.5*bar_width, true_coefficients, bar_width, label='True Coefficients', color='skyblue')
plt.bar(index - 0.5*bar_width, ridge_coefficients, bar_width, label=f'Ridge (a={ridge_alpha})', color='lightgreen', alpha=0.8)
plt.bar(index + 0.5*bar_width, lasso_coefficients, bar_width, label=f'Lasso (a={lasso_alpha})', color='lightcoral', alpha=0.8)
plt.bar(index + 1.5*bar_width, enet_coefficients, bar_width, label=f'ElasticNet (a={enet_alpha}, l1r={enet_l1_ratio})', color='salmon')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title(f"Coefficient Comparison: All Models (using default alphas)")
plt.xticks(index)
plt.legend(fontsize='small')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '07_coefficients_ElasticNet_vs_All.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved plot comparing ElasticNet coefficients: {plot_path}")
print("ElasticNet often provides a balance between Ridge's shrinkage and Lasso's feature selection.")

# Evaluate Elastic Net on the SCALED test set.
print("\nEvaluating ElasticNet model on the SCALED test set...")
y_pred_enet = enet_model.predict(X_test_scaled)
mse_enet = mean_squared_error(y_test, y_pred_enet)
r2_enet = r2_score(y_test, y_pred_enet)

print(f"  ElasticNet Mean Squared Error (MSE): {mse_enet:.4f}")
print(f"  ElasticNet R-squared (R2 Score): {r2_enet:.4f}")
print(f"  Comparison with Lasso: MSE changed from {mse_lasso:.4f} to {mse_enet:.4f}")
print(f"                     R2 changed from {r2_lasso:.4f} to {r2_enet:.4f}")

# Visualize ElasticNet predictions
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_enet, alpha=0.7, edgecolors='k', label='ElasticNet Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred_enet)")
plt.title(f"ElasticNet (a={enet_alpha}, l1r={enet_l1_ratio}): Actual vs. Predicted\nMSE: {mse_enet:.2f}, R2: {r2_enet:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plot_path = os.path.join(output_dir, '08_predictions_ElasticNet.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved scatter plot of ElasticNet predictions: {plot_path}")


# --- 5. Cross-Validation ---

print("\n--- Part 5: Cross-Validation for Robust Performance Estimation ---")

# So far, I've evaluated models on a single train/test split. The results might depend
# heavily on *which* data points ended up in the test set.
# Cross-validation (CV) gives a more reliable estimate of how the model is likely to perform
# on unseen data in general.
# K-Fold CV is common: Split the *training* data into 'k' folds. Train on k-1 folds,
# test on the remaining fold. Repeat k times, using each fold as the test set once.
# Then average the performance scores.

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
print(f"\nSetting up {k_folds}-Fold Cross-Validation.")

# --- 5a. Cross-Validation for Standard Linear Regression ---
# I'll use the original *unscaled* data for standard LR here, just to be consistent
# with how I first introduced it. For regularized models, I'd use scaled data.
print("\n--- 5a. CV for Standard Linear Regression (Unscaled Data) ---")
# Note: scoring='neg_mean_squared_error' because sklearn maximises scores,
# so we use the negative MSE (higher negative MSE is better, meaning lower positive MSE).
cv_scores_lr = cross_val_score(LinearRegression(), X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

# Convert scores back to positive MSE and calculate mean/std dev
cv_mse_lr = -cv_scores_lr # Make MSE positive
mean_cv_mse_lr = np.mean(cv_mse_lr)
std_cv_mse_lr = np.std(cv_mse_lr)

print(f"Cross-validation MSE scores for Standard LR (on {k_folds} folds):")
for i, score in enumerate(cv_mse_lr):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Average CV MSE for Standard LR: {mean_cv_mse_lr:.4f}")
print(f"  Std Dev of CV MSE for Standard LR: {std_cv_mse_lr:.4f}")
print(f"\nCompare this average CV MSE ({mean_cv_mse_lr:.4f}) to the single test set MSE ({mse_lr:.4f}).")
print("The CV average gives a more stable estimate of performance.")
print("The standard deviation tells me how much the performance varied across different folds.")


# --- 5b. Cross-Validation for Ridge Regression ---
# Now let's do CV for Ridge, using the SCALED data and the same alpha=1.0 as before.
print("\n--- 5b. CV for Ridge Regression (Scaled Data, alpha=1.0) ---")
ridge_model_for_cv = Ridge(alpha=ridge_alpha, random_state=random_seed) # Use the same alpha
cv_scores_ridge = cross_val_score(ridge_model_for_cv, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')

cv_mse_ridge = -cv_scores_ridge
mean_cv_mse_ridge = np.mean(cv_mse_ridge)
std_cv_mse_ridge = np.std(cv_mse_ridge)

print(f"Cross-validation MSE scores for Ridge (alpha={ridge_alpha}, on {k_folds} folds):")
for i, score in enumerate(cv_mse_ridge):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Average CV MSE for Ridge: {mean_cv_mse_ridge:.4f}")
print(f"  Std Dev of CV MSE for Ridge: {std_cv_mse_ridge:.4f}")
print(f"\nComparing average CV MSE: Standard LR ({mean_cv_mse_lr:.4f}) vs Ridge ({mean_cv_mse_ridge:.4f})")
print("This comparison using CV is more reliable than comparing single test set scores.")


# --- 6. Hyperparameter Tuning with Cross-Validation ---

print("\n--- Part 6: Hyperparameter Tuning using Cross-Validation ---")

# Regularized models have hyperparameters (like alpha). How do I choose the best value?
# I can use cross-validation for this! Scikit-learn has built-in models like
# RidgeCV, LassoCV, ElasticNetCV that automatically search for the best alpha using CV.

# --- 6a. Tuning Alpha for Ridge (RidgeCV) ---
print("\n--- 6a. Finding the best Alpha for Ridge using RidgeCV ---")
# Define a range of alphas to test. Logspace is often good for regularization parameters.
alphas_to_test = np.logspace(-3, 3, 100) # Test 100 alphas from 0.001 to 1000
print(f"Will test {len(alphas_to_test)} alpha values ranging from {alphas_to_test.min():.3f} to {alphas_to_test.max():.0f}.")

# RidgeCV performs K-Fold CV for each alpha and chooses the one with the best average score.
# It uses Generalized Cross-Validation (GCV) by default, which is an efficient shortcut,
# but I can specify a KFold object too. Let's stick to default GCV here for simplicity.
# Note: RidgeCV automatically fits the model with the best alpha found on the *entire* training set.
ridge_cv_model = RidgeCV(alphas=alphas_to_test, store_cv_values=True) # store_cv_values lets us see the errors for each alpha
# Fitting on SCALED data
ridge_cv_model.fit(X_train_scaled, y_train)

best_alpha_ridge = ridge_cv_model.alpha_
print(f"\nRidgeCV found the best alpha to be: {best_alpha_ridge:.4f}")

# Let's see the coefficients from this tuned Ridge model
ridge_cv_coefficients = ridge_cv_model.coef_
ridge_cv_intercept = ridge_cv_model.intercept_
print(f"\nTuned Ridge Learned Intercept: {ridge_cv_intercept:.4f}")
print(f"Tuned Ridge Learned Coefficients (best alpha={best_alpha_ridge:.4f}):")
for i, coef in enumerate(ridge_cv_coefficients):
    print(f"  Feature {i}: {coef:.4f}")

# Compare coefficients: Untuned Ridge vs Tuned Ridge
plt.figure(figsize=(12, 7))
bar_width = 0.35
index = np.arange(n_features)
plt.bar(index - bar_width/2, ridge_coefficients, bar_width, label=f'Ridge Coefs (alpha={ridge_alpha})', color='lightcoral', alpha=0.8)
plt.bar(index + bar_width/2, ridge_cv_coefficients, bar_width, label=f'Tuned Ridge Coefs (alpha={best_alpha_ridge:.4f})', color='salmon')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title(f"Coefficient Comparison: Untuned vs. Tuned Ridge (found via CV)")
plt.xticks(index)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '09_coefficients_Ridge_Tuned_vs_Untuned.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved plot comparing tuned vs untuned Ridge coefficients: {plot_path}")
print("The coefficients might change depending on whether the default alpha or the CV-tuned alpha was better.")

# Evaluate the tuned Ridge model on the SCALED test set
print("\nEvaluating the Tuned Ridge model (best alpha) on the SCALED test set...")
y_pred_ridge_cv = ridge_cv_model.predict(X_test_scaled)
mse_ridge_cv = mean_squared_error(y_test, y_pred_ridge_cv)
r2_ridge_cv = r2_score(y_test, y_pred_ridge_cv)

print(f"  Tuned Ridge Mean Squared Error (MSE): {mse_ridge_cv:.4f}")
print(f"  Tuned Ridge R-squared (R2 Score): {r2_ridge_cv:.4f}")
print(f"  Comparison with Untuned Ridge (alpha={ridge_alpha}): MSE changed from {mse_ridge:.4f} to {mse_ridge_cv:.4f}")
print(f"                                          R2 changed from {r2_ridge:.4f} to {r2_ridge_cv:.4f}")
print("Ideally, tuning alpha with CV should lead to better (or at least not worse) performance on the test set.")

# Visualize Tuned Ridge predictions
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_ridge_cv, alpha=0.7, edgecolors='k', label='Tuned Ridge Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred_ridge_cv)")
plt.title(f"Tuned Ridge Regression (best alpha={best_alpha_ridge:.2f}): Actual vs. Predicted\nMSE: {mse_ridge_cv:.2f}, R2: {r2_ridge_cv:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plot_path = os.path.join(output_dir, '10_predictions_Ridge_Tuned.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved scatter plot of Tuned Ridge predictions: {plot_path}")


# --- 6b. Tuning Alpha for Lasso (LassoCV) ---
# Let's do the same for Lasso. LassoCV usually needs explicit CV folds.
print("\n--- 6b. Finding the best Alpha for Lasso using LassoCV ---")
# Define alphas (Lasso often needs smaller alphas tested than Ridge)
lasso_alphas_to_test = np.logspace(-4, 1, 100) # e.g., 0.0001 to 10
print(f"Will test {len(lasso_alphas_to_test)} alpha values for Lasso ranging from {lasso_alphas_to_test.min():.4f} to {lasso_alphas_to_test.max():.1f}.")

# LassoCV requires explicit CV folds
lasso_cv_model = LassoCV(alphas=lasso_alphas_to_test, cv=kf, random_state=random_seed, max_iter=10000)
# Fitting on SCALED data
lasso_cv_model.fit(X_train_scaled, y_train)

best_alpha_lasso = lasso_cv_model.alpha_
print(f"\nLassoCV found the best alpha to be: {best_alpha_lasso:.4f}")

# Coefficients from tuned Lasso
lasso_cv_coefficients = lasso_cv_model.coef_
lasso_cv_intercept = lasso_cv_model.intercept_
print(f"\nTuned Lasso Learned Intercept: {lasso_cv_intercept:.4f}")
print(f"Tuned Lasso Learned Coefficients (best alpha={best_alpha_lasso:.4f}):")
num_zero_coefs_lasso_cv = np.sum(np.abs(lasso_cv_coefficients) < 1e-6)
for i, coef in enumerate(lasso_cv_coefficients):
    print(f"  Feature {i}: {coef:.4f} {'(ZEROED OUT!)' if np.abs(coef) < 1e-6 else ''}")
print(f"\nTuned Lasso forced {num_zero_coefs_lasso_cv} coefficients to zero.")

# Compare coefficients: Untuned Lasso vs Tuned Lasso
plt.figure(figsize=(12, 7))
bar_width = 0.35
index = np.arange(n_features)
plt.bar(index - bar_width/2, lasso_coefficients, bar_width, label=f'Lasso Coefs (alpha={lasso_alpha})', color='lightcoral', alpha=0.8)
plt.bar(index + bar_width/2, lasso_cv_coefficients, bar_width, label=f'Tuned Lasso Coefs (alpha={best_alpha_lasso:.4f})', color='salmon')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title(f"Coefficient Comparison: Untuned vs. Tuned Lasso (found via CV)")
plt.xticks(index)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '11_coefficients_Lasso_Tuned_vs_Untuned.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved plot comparing tuned vs untuned Lasso coefficients: {plot_path}")
print("The number of zeroed-out coefficients might change significantly based on the tuned alpha.")

# Evaluate the tuned Lasso model on the SCALED test set
print("\nEvaluating the Tuned Lasso model (best alpha) on the SCALED test set...")
y_pred_lasso_cv = lasso_cv_model.predict(X_test_scaled)
mse_lasso_cv = mean_squared_error(y_test, y_pred_lasso_cv)
r2_lasso_cv = r2_score(y_test, y_pred_lasso_cv)

print(f"  Tuned Lasso Mean Squared Error (MSE): {mse_lasso_cv:.4f}")
print(f"  Tuned Lasso R-squared (R2 Score): {r2_lasso_cv:.4f}")
print(f"  Comparison with Untuned Lasso (alpha={lasso_alpha}): MSE changed from {mse_lasso:.4f} to {mse_lasso_cv:.4f}")
print(f"                                         R2 changed from {r2_lasso:.4f} to {r2_lasso_cv:.4f}")

# Visualize Tuned Lasso predictions
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_lasso_cv, alpha=0.7, edgecolors='k', label='Tuned Lasso Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred_lasso_cv)")
plt.title(f"Tuned Lasso Regression (best alpha={best_alpha_lasso:.4f}): Actual vs. Predicted\nMSE: {mse_lasso_cv:.2f}, R2: {r2_lasso_cv:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plot_path = os.path.join(output_dir, '12_predictions_Lasso_Tuned.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved scatter plot of Tuned Lasso predictions: {plot_path}")


# --- 7. Final Comparison ---
print("\n--- Part 7: Final Summary and Comparison ---")

print("\nSummary of Test Set Performance (Lower MSE is better, Higher R2 is better):")
print("------------------------------------------------------------")
print(f"Model                      |      MSE |       R2")
print("------------------------------------------------------------")
print(f"Standard Linear Regression | {mse_lr:8.4f} | {r2_lr:8.4f}   (Unscaled Data)")
print(f"Ridge (alpha={ridge_alpha:.1f})          | {mse_ridge:8.4f} | {r2_ridge:8.4f}   (Scaled Data)")
print(f"Lasso (alpha={lasso_alpha:.1f})          | {mse_lasso:8.4f} | {r2_lasso:8.4f}   (Scaled Data)")
print(f"ElasticNet (a={enet_alpha:.1f},l1r={enet_l1_ratio:.1f}) | {mse_enet:8.4f} | {r2_enet:8.4f}   (Scaled Data)")
print("------------------------------------------------------------")
print(f"Tuned Ridge (best a={best_alpha_ridge:.3f}) | {mse_ridge_cv:8.4f} | {r2_ridge_cv:8.4f}   (Scaled Data, CV Tuned)")
print(f"Tuned Lasso (best a={best_alpha_lasso:.4f}) | {mse_lasso_cv:8.4f} | {r2_lasso_cv:8.4f}   (Scaled Data, CV Tuned)")
print("------------------------------------------------------------")

print("\nSummary of Average Cross-Validation MSE (Lower is better):")
print("-----------------------------------------------------")
print(f"Model                     | Avg CV MSE | StdDev CV MSE")
print("-----------------------------------------------------")
print(f"Standard Linear Regression| {mean_cv_mse_lr:10.4f} | {std_cv_mse_lr:10.4f} (Unscaled)")
print(f"Ridge (alpha={ridge_alpha:.1f})         | {mean_cv_mse_ridge:10.4f} | {std_cv_mse_ridge:10.4f} (Scaled)")
# Note: CV scores for tuned models aren't directly comparable as they were *used* for tuning.
# The test set scores above are the final evaluation metric for the tuned models.
print("-----------------------------------------------------")

# Final coefficient comparison plot
plt.figure(figsize=(15, 8))
bar_width = 0.15
index = np.arange(n_features)
plt.bar(index - 2.5*bar_width, true_coefficients, bar_width, label='True Coefs', color='black')
plt.bar(index - 1.5*bar_width, lr_coefficients, bar_width, label='LR Coefs (Unscaled)', color='gray', alpha=0.7)
plt.bar(index - 0.5*bar_width, ridge_cv_coefficients, bar_width, label=f'Tuned Ridge (a={best_alpha_ridge:.3f})', color='blue', alpha=0.8)
plt.bar(index + 0.5*bar_width, lasso_cv_coefficients, bar_width, label=f'Tuned Lasso (a={best_alpha_lasso:.4f})', color='red', alpha=0.8)
# Add untuned ElasticNet for context
plt.bar(index + 1.5*bar_width, enet_coefficients, bar_width, label=f'ElasticNet (a={enet_alpha:.1f},l1r={enet_l1_ratio:.1f})', color='green', alpha=0.7)

plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Final Coefficient Comparison (Incl. Tuned Models on Scaled Data)")
plt.xticks(index)
plt.legend(fontsize='small', loc='best')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5) # Add horizontal line at zero
plot_path = os.path.join(output_dir, '13_coefficients_Final_Comparison.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved final coefficient comparison plot: {plot_path}")
print("\nThis final plot shows how the coefficients differ across the methods, especially after tuning alpha.")
print("Regularization (Ridge, Lasso, ElasticNet) helps control coefficient sizes, preventing overfitting.")
print("Lasso and ElasticNet can perform feature selection by setting coefficients to zero.")
print("Cross-validation provides a robust way to estimate performance and tune hyperparameters like alpha.")

print("\n\n--- Experiment Complete ---")
print(f"All plots saved in the '{output_dir}' folder.")