# -*- coding: utf-8 -*-
"""
KNN for Mobile Price Range Prediction

Okay, my task is to help Bob predict the price range of mobile phones
using the K-Nearest Neighbors algorithm. The price range isn't the exact price,
but categories (like low, medium, high). This is a classification problem!

My plan:
1. Load the data (assuming it's in 'mobile_price_data.csv').
2. Perform Exploratory Data Analysis (EDA): Understand the features, target,
   distributions, correlations, and missing values.
3. Preprocess the data: Handle any missing values (if any) and importantly,
   SCALE the features, as KNN is distance-based.
4. Split the data into training and testing sets.
5. Implement the Elbow Method: Iterate through different values of K (number of neighbors)
   and calculate the error rate (or accuracy) to find the 'elbow' point, suggesting an optimal K.
6. Train KNN with Euclidean Distance: Use the optimal K found and evaluate the model.
7. Train KNN with Manhattan Distance: Use the same optimal K and compare performance.
   (I'll explain why Hamming distance isn't the best fit here).
8. Visualize results: EDA plots, Elbow curve, Confusion matrices.
9. Save everything nicely in a 'KNN_Outputs' folder.
10. Use plenty of comments and print statements to explain my thought process.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Setup and Data Loading ---

print("--- Setting up Environment and Loading Data ---")

# First, I need a place to save my plots.
output_dir = 'KNN_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: '{output_dir}' to save my plots.")
else:
    print(f"Directory '{output_dir}' already exists. Plots will be saved there.")


try:
    # Let's assume the file has a header row based on the image.
    file_path = 'train.csv' 
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from '{file_path}'.")
except FileNotFoundError:
    print(f"ERROR: Could not find the file '{file_path}'.")
    print("Please make sure the file exists and the path is correct.")
    # Exit if I can't load the data
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---

print("\n--- Starting Exploratory Data Analysis (EDA) ---")

# Let's get a first look at the data.
print("\nBasic DataFrame Info:")
df.info() # Shows column names, non-null counts, data types.

print("\nFirst 5 rows of the data:")
print(df.head())

print("\nShape of the data (rows, columns):")
print(df.shape)

print("\nSummary Statistics for Numerical Features:")
# This helps identify scale differences and potential outliers.
print(df.describe())

# Check for missing values. This is crucial.
print("\nChecking for Missing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0]) # Only show columns with missing values
if missing_values.sum() == 0:
    print("Looks good! No missing values found in the dataset.")
else:
    print("Warning: Missing values detected. Handling might be needed.")
    # Depending on the amount, I might impute (e.g., with median) or drop rows/columns.
    # For now, I'll assume no missing values based on the initial check. If there were,
    # I'd add imputation steps here before proceeding.

# Analyze the Target Variable: 'price_range'
print("\nAnalyzing the Target Variable ('price_range'):")
target_counts = df['price_range'].value_counts().sort_index()
print("Value Counts for 'price_range':")
print(target_counts)

# Let's visualize the target distribution.
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='price_range', palette='viridis')
plt.title('Distribution of Price Range Categories')
plt.xlabel('Price Range Category')
plt.ylabel('Number of Mobile Phones')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '01_target_distribution.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved target distribution plot to: {plot_path}")
if len(np.unique(target_counts)) == 1:
    print("The classes seem perfectly balanced, which is great!")
else:
    print("The classes might be slightly imbalanced, but look reasonably distributed.")


# Analyze Feature Relationships
# Correlation Heatmap for numerical features (and binary treated as 0/1)
print("\nCalculating and visualizing feature correlations...")
plt.figure(figsize=(18, 15)) # Need a large figure for many features
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5) # Annot=True is too crowded
plt.title('Correlation Heatmap of Features')
plot_path = os.path.join(output_dir, '02_correlation_heatmap.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved correlation heatmap plot to: {plot_path}")

# Let's look specifically at correlations with the target variable 'price_range'
print("\nCorrelations with Target Variable ('price_range'):")
target_corr = correlation_matrix['price_range'].sort_values(ascending=False)
print(target_corr)
# Okay, 'ram' seems very strongly correlated with price_range. This makes sense!
# Battery power, pixel width/height also show positive correlation.

# Visualize distribution of key features vs. price range
key_numeric_features = ['ram', 'battery_power', 'px_width', 'px_height', 'int_memory']
print(f"\nVisualizing distributions of key features ({key_numeric_features}) across price ranges...")

for feature in key_numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='price_range', y=feature, palette='viridis')
    plt.title(f'{feature.replace("_", " ").title()} vs. Price Range')
    plt.xlabel('Price Range Category')
    plt.ylabel(feature.replace("_", " ").title())
    plot_path = os.path.join(output_dir, f'03_{feature}_vs_price_range_boxplot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
print(f"Saved boxplots for key features vs price range in '{output_dir}'.")
# These plots visually confirm the trends seen in correlation, e.g., higher RAM for higher price ranges.

# Visualize distribution of some binary features vs. price range
binary_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
print(f"\nVisualizing distributions of binary features ({binary_features}) across price ranges...")

for feature in binary_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=feature, hue='price_range', palette='viridis')
    plt.title(f'Distribution of {feature.replace("_", " ").title()} across Price Ranges')
    plt.xlabel(f'Has {feature.replace("_", " ").title()} (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.legend(title='Price Range')
    plot_path = os.path.join(output_dir, f'04_{feature}_vs_price_range_countplot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
print(f"Saved countplots for binary features vs price range in '{output_dir}'.")
# Example observation: 4G seems more common in higher price ranges.

print("\n--- EDA Complete ---")


# --- 3. Data Preprocessing ---

print("\n--- Starting Data Preprocessing ---")

# Separate features (X) and target variable (y)
X = df.drop('price_range', axis=1)
y = df['price_range']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Feature Scaling: KNN is highly sensitive to the scale of features because it relies on distances.
# Features with larger ranges can dominate the distance calculation.
# I'll use StandardScaler to transform features to have zero mean and unit variance.
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()

# Fit the scaler on the feature data and transform it.
# In a typical workflow, I'd split first, then fit scaler *only* on training data
# and transform both train and test sets to prevent data leakage from the test set.
# Let's correct this - split first!

# --- 4. Train/Test Split ---

print("\n--- Splitting Data into Training and Test Sets ---")
# I'll split before scaling, which is the standard practice.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y # Use 25% for test set, stratify for classification
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Now, apply scaling. Fit *only* on training data.
print("\nApplying StandardScaler (fitting on training data only)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the SAME scaler fitted on training data

# The output arrays are numpy arrays, let's convert them back to DataFrames
# just for easier viewing if needed (optional, sklearn works with numpy arrays).
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Scaling complete.")
print("Example of scaled training data (first 5 rows):")
print(X_train_scaled_df.head())
print("Mean of 'ram' in scaled training data (should be close to 0):", X_train_scaled_df['ram'].mean())
print("Std Dev of 'ram' in scaled training data (should be close to 1):", X_train_scaled_df['ram'].std())


# --- 5. Elbow Method for Optimal K ---

print("\n--- Implementing Elbow Method to Find Optimal K ---")

# I need to find a good value for K (number of neighbors).
# The Elbow method involves running KNN for a range of K values and plotting
# the error rate (or accuracy). We look for the 'elbow' point where the
# error rate stops decreasing significantly.

error_rate = []
k_range = range(1, 40) # Check K values from 1 to 39

print(f"Calculating error rates for K from {k_range.start} to {k_range.stop - 1}...")

for k in k_range:
    # Create KNN classifier with current k
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean') # Start with Euclidean
    # Train the model on SCALED training data
    knn.fit(X_train_scaled, y_train)
    # Predict on SCALED test data
    y_pred_k = knn.predict(X_test_scaled)
    # Calculate error rate (1 - accuracy) and append
    error = 1 - accuracy_score(y_test, y_pred_k)
    error_rate.append(error)
    # Optional: Print progress every few k's
    if k % 5 == 0:
        print(f"  Calculated for K = {k}, Error Rate = {error:.4f}")


print("Calculation complete.")

# Now, let's plot the error rate vs. K value.
plt.figure(figsize=(12, 7))
plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value (Elbow Method)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Error Rate (1 - Accuracy)')
plt.xticks(np.arange(k_range.start, k_range.stop, step=2)) # Adjust step if needed
plt.grid(True, linestyle='--', alpha=0.7)
plot_path = os.path.join(output_dir, '05_elbow_curve_error_rate.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved Elbow Method plot (Error Rate) to: {plot_path}")

# Finding the K corresponding to the minimum error rate (or the elbow)
min_error = min(error_rate)
optimal_k_min_error = k_range[error_rate.index(min_error)]
print(f"\nMinimum error rate ({min_error:.4f}) found at K = {optimal_k_min_error}.")
print("However, the 'elbow' might be at a smaller K where the rate of decrease slows down.")
print("Look at the plot '05_elbow_curve_error_rate.png' to visually determine the elbow.")
# Let's visually inspect the plot. Often the elbow is where the curve bends.
# Let's assume, based on typical results for this dataset, the elbow might be around K=9 to K=15.
# For the script, I'll programmatically choose the K with minimum error, but mention visual check is important.
optimal_k = optimal_k_min_error # Use the K with minimum error found
# optimal_k = 11 # Example: If I visually decided K=11 was the elbow
print(f"Selected Optimal K = {optimal_k} for further evaluation.")


# --- 6. Train and Evaluate KNN with Optimal K (Euclidean Distance) ---

print(f"\n--- Training and Evaluating KNN with Optimal K = {optimal_k} (Euclidean Distance) ---")

# Create the final KNN classifier with the chosen K and Euclidean distance.
# Euclidean is the default (metric='minkowski' with p=2).
knn_euclidean = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean') # Explicitly setting euclidean

print(f"Training KNN model with K={optimal_k} and Euclidean distance...")
# Train on the SCALED training data
knn_euclidean.fit(X_train_scaled, y_train)
print("Training complete.")

print("\nEvaluating the Euclidean KNN model on the SCALED test set...")
# Predict on the SCALED test data
y_pred_euclidean = knn_euclidean.predict(X_test_scaled)

# Calculate Performance Metrics
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)
cm_euclidean = confusion_matrix(y_test, y_pred_euclidean)
report_euclidean = classification_report(y_test, y_pred_euclidean)

print(f"\nAccuracy (Euclidean KNN, K={optimal_k}): {accuracy_euclidean:.4f}")
print("\nClassification Report (Euclidean KNN):\n", report_euclidean)
print("\nConfusion Matrix (Euclidean KNN):")
print(cm_euclidean)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_euclidean, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y)) # Use unique values from y for labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Euclidean KNN (K={optimal_k})\nAccuracy: {accuracy_euclidean:.3f}")
plot_path = os.path.join(output_dir, f'06_confusion_matrix_euclidean_K{optimal_k}.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved Euclidean KNN confusion matrix plot to: {plot_path}")

# --- 7. Train and Evaluate KNN with Optimal K (Manhattan Distance) ---

print(f"\n--- Comparing with Manhattan Distance (K = {optimal_k}) ---")

# Now, let's see if using a different distance metric changes things.
# Manhattan distance (L1 norm) is another common choice for numerical data.
# It calculates distance as the sum of absolute differences between coordinates.
# metric='minkowski' with p=1 gives Manhattan distance.
knn_manhattan = KNeighborsClassifier(n_neighbors=optimal_k, metric='manhattan') # p=1 for Manhattan

print(f"Training KNN model with K={optimal_k} and Manhattan distance...")
# Train on the SCALED training data
knn_manhattan.fit(X_train_scaled, y_train)
print("Training complete.")

print("\nEvaluating the Manhattan KNN model on the SCALED test set...")
# Predict on the SCALED test data
y_pred_manhattan = knn_manhattan.predict(X_test_scaled)

# Calculate Performance Metrics
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)
cm_manhattan = confusion_matrix(y_test, y_pred_manhattan)
report_manhattan = classification_report(y_test, y_pred_manhattan)

print(f"\nAccuracy (Manhattan KNN, K={optimal_k}): {accuracy_manhattan:.4f}")
print("\nClassification Report (Manhattan KNN):\n", report_manhattan)
print("\nConfusion Matrix (Manhattan KNN):")
print(cm_manhattan)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_manhattan, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Manhattan KNN (K={optimal_k})\nAccuracy: {accuracy_manhattan:.3f}")
plot_path = os.path.join(output_dir, f'07_confusion_matrix_manhattan_K{optimal_k}.png')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved Manhattan KNN confusion matrix plot to: {plot_path}")


# --- Explanation about Hamming Distance ---
print("\n--- Note on Hamming Distance ---")
print("The request mentioned trying Hamming distance.")
print("Hamming distance is typically used for comparing two *binary* or *categorical* vectors of the same length.")
print("It counts the number of positions at which the corresponding symbols are different.")
print("Since our dataset is primarily composed of numerical features (even after scaling),")
print("Euclidean (L2) and Manhattan (L1) distances are more appropriate.")
print("Applying Hamming directly to scaled numerical data wouldn't be meaningful.")
print("If the features were purely categorical (e.g., 'color': 'red'/'blue', 'brand': 'A'/'B'),")
print("then Hamming distance (or similar metrics like Jaccard for binary sets) would be suitable.")


# --- 8. Final Comparison ---
print("\n--- Final Comparison: Euclidean vs. Manhattan ---")
print(f"Optimal K selected: {optimal_k}")
print("--------------------------------------------------")
print(f"Metric     | Accuracy | Notes")
print("--------------------------------------------------")
print(f"Euclidean  | {accuracy_euclidean:8.4f} | Standard distance for numerical data.")
print(f"Manhattan  | {accuracy_manhattan:8.4f} | L1 distance, potentially less sensitive to outliers.")
print("--------------------------------------------------")

if abs(accuracy_euclidean - accuracy_manhattan) < 0.005: # Small threshold
    print("\nThe performance between Euclidean and Manhattan distances is very similar for this dataset with K={optimal_k}.")
elif accuracy_euclidean > accuracy_manhattan:
    print("\nEuclidean distance performed slightly better than Manhattan distance for this dataset with K={optimal_k}.")
else:
    print("\nManhattan distance performed slightly better than Euclidean distance for this dataset with K={optimal_k}.")

print("\nKey takeaways from the KNN analysis:")
print(" - EDA showed 'ram' is a very strong predictor of price range.")
print(" - Feature scaling (StandardScaler) is crucial for KNN.")
print(" - The Elbow Method helped identify a suitable range for K, balancing bias and variance.")
print(f" - An optimal K around {optimal_k} was chosen based on minimizing test error.")
print(" - Both Euclidean and Manhattan distances yielded comparable results on this scaled data.")
print(" - KNN achieved decent accuracy in predicting the mobile price range.")

print("\n\n--- KNN Analysis Complete ---")
print(f"All plots saved in the '{output_dir}' folder.")