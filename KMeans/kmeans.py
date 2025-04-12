# -*- coding: utf-8 -*-
"""
KMeans Clustering Analysis with Optimal K Determination

Alright, my mission is to apply KMeans clustering to a bunch of 2D datasets.
The main goals are:
1. Load each dataset (.csv file).
2. Perform some basic Exploratory Data Analysis (EDA) - mostly visualization.
3. Determine the 'optimal' number of clusters (K) for each dataset using:
    a. The Elbow Method (based on inertia/WCSS).
    b. The Silhouette Score method.
4. Apply KMeans clustering using the determined optimal K.
5. Visualize the original data, the K determination process, and the final clustering results.
6. Save all generated plots into a dedicated folder.

I'll be using scikit-learn's KMeans implementation. Let's get started!
"""

import os
import glob # To find all the csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Important for KMeans!
from sklearn.metrics import silhouette_score # For finding optimal K

# --- 1. Setup Environment ---

print("--- Setting up Environment ---")

# Define the directory where the data files are located
# Assuming the script is run from the directory containing the CSV files
data_dir = '.' # Current directory

# Define the directory for saving plots
output_dir = 'KMeans_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: '{output_dir}' to save my plots.")
else:
    print(f"Directory '{output_dir}' already exists. Plots will be saved there.")

# Find all CSV files in the data directory
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
print(f"\nFound {len(csv_files)} CSV files to process:")
# Sort files for consistent processing order
csv_files.sort()
for f in csv_files:
    print(f" - {os.path.basename(f)}")

# Define a range of K values to test for optimal K determination
# Going up to 10 clusters should be sufficient for most of these datasets, maybe 15 for complex ones.
# Let's check visually and adjust if needed. A range 2-15 seems reasonable.
k_range = range(2, 16)

# --- 2. Process Each Dataset ---

print("\n--- Starting Dataset Processing Loop ---")

# Check if any files were found
if not csv_files:
    print("\nERROR: No CSV files found in the current directory.")
    print("Please ensure the script is in the same directory as the data files.")
    exit()

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0] # Get filename without extension

    print(f"\n\n================ Processing: {file_name} ================")

    # --- 2.1 Load Data ---
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from '{file_name}'. Shape: {df.shape}")
        # Display first few rows and info
        print("First 5 rows:")
        print(df.head())
        print("\nData Info:")
        df.info()
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print("\nWarning: Found missing values!")
            print(df.isnull().sum())
            # For simplicity here, I'll drop rows with NaNs if any.
            # In a real scenario, more sophisticated imputation might be needed.
            df.dropna(inplace=True)
            print(f"Dropped rows with NaNs. New shape: {df.shape}")
        else:
            print("\nNo missing values found. Good.")

    except Exception as e:
        print(f"ERROR: Could not load or process {file_name}. Skipping. Error: {e}")
        continue # Skip to the next file

    # --- 2.2 Prepare Features (X) ---
    # Assuming columns are 'x', 'y', and optionally 'color' based on context
    if 'x' not in df.columns or 'y' not in df.columns:
        print(f"ERROR: Required columns 'x' and 'y' not found in {file_name}. Skipping.")
        continue

    # Features for clustering are just the coordinates
    X = df[['x', 'y']]
    print(f"\nExtracted features (X) with shape: {X.shape}")

    # Keep track of the 'ground truth' color if it exists, for visualization later
    ground_truth_colors = None
    if 'color' in df.columns:
        ground_truth_colors = df['color']
        print("Found 'color' column, will use it for visualizing ground truth.")
        # How many true clusters are there?
        n_true_clusters = ground_truth_colors.nunique()
        print(f"Number of unique 'colors' (potential true clusters): {n_true_clusters}")


    # --- 2.3 Visualize Raw Data ---
    plt.figure(figsize=(8, 6))
    # If we have ground truth colors, use them! Otherwise, just plot points.
    if ground_truth_colors is not None:
        unique_colors = ground_truth_colors.unique()
        # Use a palette that can handle a decent number of categories
        palette = sns.color_palette("viridis", n_colors=len(unique_colors))
        sns.scatterplot(x=X['x'], y=X['y'], hue=ground_truth_colors, palette=palette, legend='full', s=50, alpha=0.7)
        plt.title(f'Raw Data: {base_name} (Colored by Ground Truth)')
        plt.legend(title='True Color/Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(x=X['x'], y=X['y'], s=50, alpha=0.7)
        plt.title(f'Raw Data: {base_name}')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path = os.path.join(output_dir, f'{base_name}_01_raw_data.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close() # Close the plot to free memory
    print(f"Saved raw data plot to: {plot_path}")


    # --- 2.4 Feature Scaling ---
    print("\nApplying StandardScaler to features (X)...")
    # KMeans uses Euclidean distance, which is sensitive to feature scales.
    # Scaling ensures that both 'x' and 'y' contribute fairly to the distance calculation.
    # StandardScaler removes the mean and scales to unit variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Scaling complete. Shape of scaled data: {X_scaled.shape}")
    # print("First 5 rows of scaled data:") # Optional: print scaled data
    # print(pd.DataFrame(X_scaled, columns=['x_scaled', 'y_scaled']).head())


    # --- 2.5 Optimal K Determination ---
    print("\nDetermining optimal K using Elbow and Silhouette methods...")
    inertia_values = []
    silhouette_scores = []

    # It's important to run KMeans multiple times for each K with different seeds
    # to ensure stability, hence n_init=10.
    # Using 'k-means++' for initialization is generally better than 'random'.
    # Setting random_state for reproducibility of the entire process.
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    print(f"Testing K values in range: {list(k_range)}")
    for k in k_range:
        # print(f"  - Fitting KMeans for K={k}...") # Can be verbose
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X_scaled)

        # 1. Inertia (WCSS) for Elbow Method
        inertia_values.append(kmeans.inertia_)

        # 2. Silhouette Score
        # Requires cluster labels, which we just got from fitting
        cluster_labels = kmeans.labels_
        # Silhouette score requires at least 2 labels.
        if k >= 2:
            score = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(np.nan) # Cannot compute for K=1

        # print(f"    K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score if k>=2 else 'N/A'}")

    # --- 2.5.1 Plot Elbow Method ---
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_values, marker='o', linestyle='-')
    plt.title(f'Elbow Method for Optimal K ({base_name})')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(list(k_range))
    plt.grid(True, linestyle='--', alpha=0.7)
    # Try to automatically find the elbow point (using KneeLocator, optional)
    try:
        from kneed import KneeLocator
        kl = KneeLocator(list(k_range), inertia_values, curve='convex', direction='decreasing')
        elbow_k = kl.elbow
        if elbow_k:
            plt.vlines(elbow_k, plt.ylim()[0], plt.ylim()[1], linestyles='--', color='r', label=f'Elbow at K={elbow_k}')
            plt.legend()
            print(f"  - Elbow method suggests K={elbow_k}")
        else:
            print("  - Elbow point not automatically detected.")
            elbow_k = None # Set to None if not found
    except ImportError:
        print("  - Elbow method: Manual inspection needed (KneeLocator not installed).")
        elbow_k = None # Set to None if KneeLocator isn't available

    plot_path = os.path.join(output_dir, f'{base_name}_02_elbow_method.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Elbow method plot to: {plot_path}")


    # --- 2.5.2 Plot Silhouette Scores ---
    plt.figure(figsize=(8, 5))
    # We only calculated scores for K>=2
    valid_k_range = [k for k in k_range if k >= 2]
    valid_silhouette_scores = [s for s in silhouette_scores if not np.isnan(s)]

    if valid_silhouette_scores: # Check if we have scores to plot
        plt.plot(valid_k_range, valid_silhouette_scores, marker='o', linestyle='-')
        plt.title(f'Silhouette Scores for Optimal K ({base_name})')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Average Silhouette Score')
        plt.xticks(valid_k_range)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Find the K with the highest silhouette score
        optimal_k_silhouette = valid_k_range[np.argmax(valid_silhouette_scores)]
        plt.vlines(optimal_k_silhouette, plt.ylim()[0], plt.ylim()[1], linestyles='--', color='g', label=f'Max Silhouette at K={optimal_k_silhouette}')
        plt.legend()
        print(f"  - Silhouette method suggests K={optimal_k_silhouette} (Score: {max(valid_silhouette_scores):.4f})")

        plot_path = os.path.join(output_dir, f'{base_name}_03_silhouette_scores.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Silhouette scores plot to: {plot_path}")
    else:
        print("  - Could not calculate Silhouette scores (maybe only K=1 was tested?).")
        optimal_k_silhouette = None # No optimal K from Silhouette


    # --- 2.6 Determine Final Optimal K and Run Final KMeans ---

    # Choosing the optimal K:
    # Often Silhouette score is more reliable than the Elbow method, especially if the elbow is ambiguous.
    # Let's prioritize the Silhouette score peak. If it's unavailable or unclear,
    # we might fall back to the Elbow K (if found) or use the 'true' K if available and looks reasonable.
    # For robust datasets like 'blob', both should ideally agree. For complex shapes like 'spirals',
    # KMeans might struggle anyway, and the optimal K might be less meaningful.

    final_k = None
    if optimal_k_silhouette:
        final_k = optimal_k_silhouette
        print(f"\nSelected K = {final_k} based on peak Silhouette score.")
    elif elbow_k:
        final_k = elbow_k
        print(f"\nSelected K = {final_k} based on Elbow method (Silhouette peak was unclear/unavailable).")
    elif n_true_clusters is not None and n_true_clusters in k_range:
         # Fallback if other methods fail, maybe use the known number of colors? Risky assumption.
         # Let's stick to the computed metrics for now.
         print("\nWarning: Could not determine K from Silhouette or Elbow. Clustering results might be suboptimal.")
         # Maybe pick a default or median K? Let's try K=3 as a guess if nothing else works.
         final_k = 3
         print(f"Defaulting to K = {final_k} as other methods failed.")
    else:
        final_k = 3 # Default fallback
        print("\nWarning: Could not determine K from Silhouette or Elbow. Defaulting to K = {final_k}.")


    if final_k is None or final_k < 2:
         print(f"ERROR: Could not determine a valid K (>= 2) for {file_name}. Skipping final clustering.")
         continue


    print(f"\nRunning final KMeans clustering with K = {final_k}...")
    final_kmeans = KMeans(n_clusters=final_k, **kmeans_kwargs)
    final_kmeans.fit(X_scaled)

    # Get the results
    final_labels = final_kmeans.labels_
    final_centroids_scaled = final_kmeans.cluster_centers_

    # We need to transform the centroids back to the original scale for plotting
    final_centroids_original = scaler.inverse_transform(final_centroids_scaled)
    print("Final KMeans fitting complete.")
    print(f"Shape of cluster labels: {final_labels.shape}")
    print(f"Shape of scaled centroids: {final_centroids_scaled.shape}")
    print("Cluster centroids (original scale):")
    print(pd.DataFrame(final_centroids_original, columns=['x_center', 'y_center']))


    # --- 2.7 Visualize Final Clustering Results ---
    plt.figure(figsize=(10, 8)) # Make it slightly larger to include centroids clearly

    # Use a palette for the identified clusters
    cluster_palette = sns.color_palette("viridis", n_colors=final_k)

    # Plot data points colored by the assigned cluster label
    sns.scatterplot(x=X['x'], y=X['y'], hue=final_labels, palette=cluster_palette, legend='full', s=50, alpha=0.7)

    # Plot the centroids on top
    plt.scatter(final_centroids_original[:, 0], final_centroids_original[:, 1],
                marker='X', s=200, c='red', edgecolors='black', label='Centroids')

    plt.title(f'KMeans Clustering Result: {base_name} (K={final_k})')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(output_dir, f'{base_name}_04_kmeans_result_k{final_k}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved final KMeans clustering plot to: {plot_path}")


    # --- 2.8 (Optional) Compare with Ground Truth if available ---
    if ground_truth_colors is not None:
         print("\nComparing KMeans result with ground truth visually (see saved plots).")
         # Quantitative comparison (like Adjusted Rand Index) could be done here too,
         # but the request focused on K determination and visualization.
         # from sklearn.metrics import adjusted_rand_score
         # ari_score = adjusted_rand_score(ground_truth_colors, final_labels)
         # print(f"Adjusted Rand Index (ARI) between KMeans labels and ground truth: {ari_score:.4f}")
         # A score close to 1 means high similarity, close to 0 means random labeling.

    print(f"================ Finished processing: {file_name} ================")


print("\n\n--- All Datasets Processed ---")
print(f"All plots saved in the '{output_dir}' folder.")