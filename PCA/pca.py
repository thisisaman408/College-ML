# -*- coding: utf-8 -*-
"""
PCA for Mobile Phone Dataset Dimensionality Reduction

Okay, this time I'm exploring Principal Component Analysis (PCA).
My main goal is to reduce the number of features (dimensions) in the mobile phone dataset.
This can be useful for speeding up models, reducing noise, and sometimes even improving performance.
PCA finds new, uncorrelated features (principal components) that capture the maximum variance.

Key Steps:
1. Load train and test data.
2. Separate features (X) and target (y) - PCA works on X only.
3. Scale the features (X_train, X_test) using StandardScaler - PCA requires this!
4. Apply PCA on scaled training data to understand explained variance.
5. Choose the number of components needed (e.g., to capture 95% variance).
6. Apply PCA again with the chosen number of components for actual reduction.
7. Transform both train and test sets to the new PCA feature space.
8. Analyze and visualize the results (explained variance, 2D projection).
9. Save plots in 'PCA_Outputs'.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. Setup and Data Loading ---

print("--- Setting up Environment and Loading Data ---")

# Directory for saving plots
output_dir = 'PCA_Outputs'
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
    df_test = pd.read_csv(test_file)
    print(f"Successfully loaded training data from '{train_file}' ({df_train.shape}).")
    print(f"Successfully loaded testing data from '{test_file}' ({df_test.shape}).")

    # Verify 'price_range' is in both, handle if test.csv is just features
    if 'price_range' not in df_test.columns:
        print("Warning: 'price_range' column not found in test.csv.")
        # If test.csv is for submission/prediction only, I might not need y_test
        # For demonstration purposes here, let's split df_train if y_test isn't available
        print("Splitting train.csv into training and validation sets (80/20) for consistent y_test.")
        df_full = df_train.copy()
        df_train, df_test = train_test_split(df_full, test_size=0.20, random_state=42, stratify=df_full['price_range'])
        print(f"New training set size: {df_train.shape}")
        print(f"New validation set size: {df_test.shape}")


except FileNotFoundError as e:
    print(f"ERROR: Could not find the data file: {e}. Please ensure 'train.csv' and 'test.csv' are present.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- 2. Prepare Data for PCA ---

print("\n--- Preparing Data for PCA ---")

# Separate features (X) and target variable (y)
# PCA is unsupervised, so it's applied only to features (X).
# I'll keep 'y' to visualize the results later.
X_train = df_train.drop('price_range', axis=1)
y_train = df_train['price_range']

X_test = df_test.drop('price_range', axis=1)
y_test = df_test['price_range'] # Assuming y_test is available based on handling above

print(f"Training features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Testing target shape: {y_test.shape}")
n_original_features = X_train.shape[1]
print(f"Number of original features: {n_original_features}")


# --- 3. Feature Scaling (Normalization) ---

print("\n--- Scaling Features using StandardScaler ---")
# PCA is sensitive to the scale of the data. Features with larger variances
# will have a disproportionately large influence on the principal components.
# So, I MUST scale the data first. StandardScaler standardizes features
# by removing the mean and scaling to unit variance.

scaler = StandardScaler()

# Fit scaler ONLY on training data to avoid data leakage
X_train_scaled = scaler.fit_transform(X_train)

# Transform both training and test data using the SAME fitted scaler
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete.")
print(f"Shape of scaled training data: {X_train_scaled.shape}")


# --- 4. Apply PCA and Analyze Explained Variance ---

print("\n--- Applying PCA to Analyze Explained Variance ---")
# Initially, I'll fit PCA without specifying the number of components.
# This way, I can see how much variance each component explains.
pca_analyzer = PCA(random_state=42) # No n_components specified yet

# Fit PCA on the SCALED training data
print("Fitting PCA on scaled training data to calculate all components...")
pca_analyzer.fit(X_train_scaled)
print("PCA fitting complete.")

# Examine the explained variance ratio
explained_variance_ratio = pca_analyzer.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print(f"\nExplained variance ratio by each component (first 5): {explained_variance_ratio[:5].round(4)}")
print(f"Cumulative explained variance (first 5 components): {cumulative_explained_variance[:5].round(4)}")

# Plot the cumulative explained variance
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='red', linestyle=':', label='95% Explained Variance')
plt.axhline(y=0.99, color='green', linestyle=':', label='99% Explained Variance')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(np.arange(0, n_original_features + 1, step=2)) # Adjust step if needed
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plot_path = os.path.join(output_dir, '01_pca_explained_variance.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved cumulative explained variance plot to: {plot_path}")
print("I need to look at this plot to decide how many components to keep.")
print("The goal is to find the 'elbow' or the point where adding more components")
print("doesn't significantly increase the explained variance, or choose a threshold (e.g., 95%).")

# Determine the number of components needed for a threshold (e.g., 95%)
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
n_components_99 = np.argmax(cumulative_explained_variance >= 0.99) + 1
print(f"\nNumber of components needed to explain:")
print(f" - At least 95% variance: {n_components_95}")
print(f" - At least 99% variance: {n_components_99}")

# Let's choose the number of components for 95% variance for dimension reduction.
n_components_chosen = n_components_95
print(f"\nI will proceed with dimension reduction using {n_components_chosen} components (for >=95% variance).")


# --- 5. Perform Dimension Reduction ---

print(f"\n--- Applying PCA for Dimension Reduction (keeping {n_components_chosen} components) ---")

# Now, instantiate PCA again with the chosen number of components
pca_reducer = PCA(n_components=n_components_chosen, random_state=42)

# Fit PCA on the SCALED training data
print(f"Fitting PCA with n_components={n_components_chosen} on scaled training data...")
pca_reducer.fit(X_train_scaled)
print("PCA fitting complete.")

# Transform both the scaled training and test data into the lower-dimensional space
print("Transforming training and test data using the fitted PCA...")
X_train_pca = pca_reducer.transform(X_train_scaled)
X_test_pca = pca_reducer.transform(X_test_scaled)

print("\nDimension Reduction Complete!")
print(f"Original training data shape: {X_train_scaled.shape}")
print(f"Reduced training data shape (PCA): {X_train_pca.shape}")
print(f"Original test data shape: {X_test_scaled.shape}")
print(f"Reduced test data shape (PCA): {X_test_pca.shape}")

# Show an example of the transformed data
print("\nFirst 5 rows of the PCA-transformed training data:")
pca_cols = [f'PC{i+1}' for i in range(n_components_chosen)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
print(X_train_pca_df.head().round(4))
print("\nNote: These Principal Components (PC1, PC2, ...) are new features.")
print("They are linear combinations of the original scaled features and are uncorrelated.")


# --- 6. Visualize PCA Results ---

print("\n--- Visualizing the PCA Transformation ---")
# Let's visualize the data projected onto the first two principal components.
# This helps see if these components capture some structure related to the price range.

plt.figure(figsize=(12, 9))
scatter = sns.scatterplot(
    x=X_train_pca[:, 0],  # First Principal Component (PC1)
    y=X_train_pca[:, 1],  # Second Principal Component (PC2)
    hue=y_train,          # Color points by the original price range
    palette='viridis',    # Color scheme
    alpha=0.7,
    s=50                  # Marker size
)
plt.title('Training Data Projected onto First Two Principal Components (PC1 vs PC2)')
plt.xlabel(f'Principal Component 1 ({pca_analyzer.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 ({pca_analyzer.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.legend(title='Price Range')
plt.grid(True, linestyle='--', alpha=0.5)
plot_path = os.path.join(output_dir, '02_pca_2d_scatter_train.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved 2D PCA scatter plot to: {plot_path}")
print("This plot shows how the different price ranges distribute along the first two principal components.")
print("If the classes are well separated here, it suggests PCA captured meaningful structure.")

# (Optional) Analyze PCA components loadings
# pca_reducer.components_ shows the relationship between original features and PCs
# print("\nLoadings for the first Principal Component (PC1):")
# loadings_pc1 = pd.Series(pca_reducer.components_[0], index=X_train.columns)
# print(loadings_pc1.sort_values(ascending=False))
# This would show which original features contribute most strongly to PC1.


# --- 7. Summary ---
print("\n--- PCA Summary ---")
print(f"Original number of features: {n_original_features}")
print(f"Features were scaled using StandardScaler.")
print(f"Number of Principal Components chosen to retain >=95% variance: {n_components_chosen}")
print(f"Reduced number of features: {X_train_pca.shape[1]}")
print(f"Total variance explained by {n_components_chosen} components: {cumulative_explained_variance[n_components_chosen-1]:.4f}")
print("\nPCA provides a way to reduce dimensionality while preserving most of the data's variance.")
print("The transformed features (principal components) are uncorrelated.")
print("This reduced dataset can now be used as input for machine learning models, potentially")
print("leading to faster training times and sometimes better generalization by reducing noise.")


print("\n\n--- PCA Analysis Complete ---")
print(f"All plots saved in the '{output_dir}' folder.")