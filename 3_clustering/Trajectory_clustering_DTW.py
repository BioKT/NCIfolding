import os
import re
import configparser
import mdtraj as md
import numpy as np
import pandas as pd
from aeon.distances import dtw_pairwise_distance
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

# Set the arial font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Parameteres used for DTW clustering
n_clusters_range = range(2, 11)
bounding_window = 0.5

# Density data used for the clustering 
density_types = ['SG Attractive n^1', 'SG VdW n^1', 'SG Repulsive n^1']

# Define residue pairs of each structural elements to track; these lines were taken from a res_pairs_tsmsm_xxx.txt 
structural_elements = {
    "beta_sheet12": [[2, 18], [4, 15], [4, 16], [3, 16], [3, 15], [2, 20], [0, 25], [3, 17], [0, 22], [1, 18], [1, 19], [1, 17], [0, 19], [4, 18], [0, 20], [2, 17]],
    "beta_sheet13": [[1, 38], [5, 36], [3, 38], [5, 34], [4, 35], [3, 36], [6, 34], [2, 35]],
    "alpha_helix1": [[25, 29], [25, 30], [21, 24], [21, 25], [22, 26], [23, 27], [20, 25], [24, 28], [24, 29]],
    "alpha_helix2": [[29, 35], [30, 33], [29, 34], [28, 31], [30, 35], [28, 32], [27, 31]],
    }

# Define colors for structural elements, we match them to the colors assigned by the residue pair clustering
structural_colors = {
    "beta_sheet12": "blue",
    "beta_sheet13": "green",
    "alpha_helix1": "orange",
    "alpha_helix2": "red",
    }

title_colors = {
    'SG Attractive n^1': 'blue',
    'SG VdW n^1': 'green',
    'SG Repulsive n^1': 'red'}

density_title = [r"$\int \rho_{\mathrm{attractive}}$", r"$\int \rho_{\mathrm{vdW}}$", r"$\int \rho_{\mathrm{repulsive}}$"]

# Load configuration from relevant_data
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
topology_file = config.get('Settings', 'topology_file')
waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)

# Prepare multiple directories for outputs 
output_dir = os.path.join(os.getcwd(), 'Clustering_results')
os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.join(output_dir, f'Trajectory_clustering_agglomerative_DTW_bounding_window_{str(bounding_window).replace('.', 'p')}')
os.makedirs(output_dir, exist_ok=True)
tsne_pca_output_dir = os.path.join(output_dir, "tSNE_and_PCA_analysis")
os.makedirs(tsne_pca_output_dir, exist_ok=True)

# Load topology and a frame to get the residues 
PDB_top = md.load(topology_file, standard_names=False).topology
prot_indexes = PDB_top.select('protein')
sample_frame = md.load_frame(topology_file, 0, top=PDB_top)
n_residues = sample_frame.atom_slice(prot_indexes).n_residues

# Create residue label mappings
index_to_label_dict = {i: str(PDB_top.residue(i))[3:].zfill(len(str(n_residues - 1))) + str(PDB_top.residue(i))[:3]    for i in range(n_residues)}
label_to_index_dict = {i: res for res, i in index_to_label_dict.items()}

# Find and sort traj directories
replica_directories = sorted(
    [d for d in os.listdir('.') if os.path.isdir(d) and ('_unfolding_' in d or '_folding_' in d)],
    key=lambda x: (int(re.search(r'event_(\d+)', x).group(1)), '_folding_' not in x))

# Collect densities from each trajectory
all_trajectories = []
for replica_dir in replica_directories:
    print(replica_dir)
    is_unfolding = '_unfolding_' in replica_dir #Independently feeed each density for each structural element
    replica_densities = {element: {density: [] for density in density_types} for element in structural_elements}

    for filename in os.listdir(os.path.join(replica_dir, 'NCI_csvs')):
        if '_densities' in filename and filename.endswith('.csv'):
            filepath = os.path.join(replica_dir, 'NCI_csvs', filename)
            pair_NCI_df = pd.read_csv(filepath)
            # Go over every csv extracting the densities of interest for the clustering
            if set(density_types).issubset(pair_NCI_df.columns):
                match = re.search(r'event_(\d+)_(\d+\w+)_vs_(\d+\w+)_densities.csv', filename)
                if match:
                    res1, res2 = match.group(2), match.group(3)
                    residue_index_pair = [label_to_index_dict[res1], label_to_index_dict[res2]]
                    # Match the residues to the labels in the file name and add them to the corresponding category in the series
                    for element, pairs in structural_elements.items():
                        if residue_index_pair in pairs:
                            for i, density in enumerate(density_types):
                                if is_unfolding:    # If its an unfolding trajectory we add the data in reverse order
                                    replica_densities[element][density].append(pair_NCI_df[density].to_numpy()[::-1])
                                else:
                                    replica_densities[element][density].append(pair_NCI_df[density].to_numpy())

    #Once all csv files have been processes the density sums are added to the total data to be clustered 
    traj_data = []
    for element in structural_elements:
        for density in density_types:
            if replica_densities[element][density]:  # Avoid empty lists
                summed_density = np.sum(replica_densities[element][density], axis=0)
            else:
                summed_density = np.zeros(1)  # Placeholder if no data available
            traj_data.append(summed_density)

    all_trajectories.append(traj_data)

# Compute max densities per density type across all trajs
max_densities = np.zeros(3*len(structural_elements))  # Placeholder for max values
for i in range(3*len(structural_elements)):
    max_densities[i] = max(np.max(traj[i]) for traj in all_trajectories if traj[i].size > 1)  # Avoid empty cases

# Normalize traj data based on global max for each density type and structural motif 
normalized_trajs = []
for traj in all_trajectories:
    normalized_traj = [np.array(traj[i]) / max_densities[i] if max_densities[i] > 0 else np.array(traj[i]) for i in range(3*len(structural_elements))]
    normalized_trajs.append(np.array(normalized_traj, dtype=np.float64))

distance_matrix = dtw_pairwise_distance(normalized_trajs, window = bounding_window)
print('distance matrix shape:',np.array(distance_matrix).shape)


# Store results for elbow and silhouette methods
elbow_inertia = []
silhouette_scores = []
davies_bouldin_scores = []
# Clustering and plotting
for n_clusters in range(2, 11):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    fig, axes = plt.subplots(3, n_clusters, figsize=(3 * n_clusters, 9), sharex=True, sharey=True)
    
    # We plot per cluster and density type
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_trajs = [normalized_trajs[i] for i in cluster_indices]

        for i, density_type in enumerate(density_types):
            for j, (element, color) in enumerate(structural_colors.items()):

                element_series = [traj[j * 3 + i] for traj in cluster_trajs]
                norm_time = [np.linspace(0, 1, len(series)) for series in element_series]
                
                # Compute average and std at aligned timepoints
                avg_density = np.mean([np.interp(np.linspace(0, 1, 100), time, series) for time, series in zip(norm_time, element_series)], axis=0)
                std_density = np.std([np.interp(np.linspace(0, 1, 100), time, series) for time, series in zip(norm_time, element_series)], axis=0)
                timepoints = np.linspace(0, 1, 100)
                
                # Plot cluster average
                normalized_time = np.linspace(0, 1, 100)  # Consistent time axis

                axes[i,cluster_id].plot(normalized_time, avg_density, label=element, color=color, linewidth=3)
                axes[i,cluster_id].fill_between(normalized_time, avg_density - std_density, avg_density + std_density, color=color, alpha=0.2)
            
            axes[i,cluster_id].set_title(f"Cluster {cluster_id + 1} {density_title[i]}", fontsize=20, color=title_colors[density_type])
            axes[i,cluster_id].set_xlim(0, 1)
            axes[i,cluster_id].set_ylim(0, 1)
            axes[i,cluster_id].tick_params(labelsize=16)

            if cluster_id == 0:
                axes[i,cluster_id].set_ylabel("Density fraction", fontsize=18)
            if i == 2:
                axes[i,cluster_id].set_xlabel("Transition length", fontsize=18)
        
        # Add traj indices as text annotation in the first column
        cluster_text = f"trajs: {', '.join(map(str, cluster_indices))}"
        axes[cluster_id, 0].text(0.02, 0.95, cluster_text, transform=axes[cluster_id, 0].transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
    

    plt.suptitle(f"Clustering DTW using bounding_window = {str(bounding_window)} with {n_clusters} Clusters", fontsize=16, fontweight="bold")
    plt.subplots_adjust(top=0.9)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"DTW_bounding_window_{str(bounding_window).replace('.', 'p')}_{n_clusters}_clusters.png"), dpi=300, bbox_inches="tight")
    plt.close()


    # Lastly I want to use the cluster labels to score the quality of the clustering before reseting them
    # Compute "inertia" (sum of squared distances within clusters)
    inertia = 0
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        inertia += np.sum(cluster_distances) / (2 * len(cluster_indices))  # Average within-cluster distance
    elbow_inertia.append(inertia)

    # Compute Silhouette Score (higher is better)
    if n_clusters > 1:  # Silhouette is only valid for n > 1
        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
        silhouette_scores.append(silhouette_avg)

    # Lastly compute Davies Bouldin score (lowe is better)
    davies_bouldin_scores.append(davies_bouldin_score(distance_matrix, cluster_labels))

# Plot Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(n_clusters_range, elbow_inertia, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Within-Cluster Distances (Inertia)")
plt.title("Elbow Method for Optimal Clusters")
plt.xticks(n_clusters_range)
plt.grid()
plt.savefig(os.path.join(output_dir, f"DTW_bounding_window_{str(bounding_window).replace('.', 'p')}_elbow_method.png"), dpi=300, bbox_inches="tight")

# Plot Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(n_clusters_range, silhouette_scores, marker='s', linestyle='-', color='g')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal Clusters")
plt.xticks(n_clusters_range)
plt.grid()
plt.savefig(os.path.join(output_dir, f"DTW_bounding_window_{str(bounding_window).replace('.', 'p')}_silhouette_analysis.png"), dpi=300, bbox_inches="tight")

# Plot Davies-Bouldin index
plt.figure(figsize=(8, 4))
plt.plot(n_clusters_range, davies_bouldin_scores, marker='^', linestyle='-', color='brown')
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Index for Clustering Quality")
plt.xticks(n_clusters_range)
plt.grid()
plt.savefig(os.path.join(output_dir, "davies_bouldin_analysis.png"), dpi=300, bbox_inches="tight")


# Compute PCA
pca = PCA(n_components=2)
pca_proj = pca.fit_transform(distance_matrix)  # PCA works directly on distance matrix

tsne = TSNE(n_components=2, metric="precomputed", init="random", perplexity=5, random_state=42)
tsne_proj = tsne.fit_transform(distance_matrix)

# Loop over clustering results
for n_clusters in n_clusters_range:
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Compute t-SNE projection (init must be 'random' for precomputed distances)
    tsne = TSNE(n_components=2, metric="precomputed", init="random", perplexity=5, random_state=42)
    tsne_proj = tsne.fit_transform(distance_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Plot PCA
    scatter_pca = axes[0].scatter(pca_proj[:, 0], pca_proj[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
    axes[0].set_title(f"PCA Clusters (n={n_clusters})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    plt.colorbar(scatter_pca, ax=axes[0])

    # Plot t-SNE
    scatter_tsne = axes[1].scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
    axes[1].set_title(f"t-SNE Clusters (n={n_clusters})")
    axes[1].set_xlabel("t-SNE1")
    axes[1].set_ylabel("t-SNE2")
    plt.colorbar(scatter_tsne, ax=axes[1])

    plt.suptitle(f"t-SNE & PCA Clustering (n={n_clusters})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(tsne_pca_output_dir, f"tSNE_PCA_n{n_clusters}.png"), dpi=300, bbox_inches="tight")
    plt.close()