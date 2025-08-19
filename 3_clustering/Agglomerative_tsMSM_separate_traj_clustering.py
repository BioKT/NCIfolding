import os
import configparser
import mdtraj as md
import numpy as np
import pandas as pd
import re
from sklearn.cluster import AgglomerativeClustering
from tslearn.utils import to_time_series_dataset
from aeon.distances import msm_pairwise_distance
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns

# Set the arial font
plt.rcParams['font.family'] = 'Arial'

# Load relevant info for analysis from the relevant_data_file
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
try:
    header = config.get('Settings', 'header')
    topology_file = config.get('Settings', 'topology_file')
    waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")

# Define the number of clusters, penalty and time window considered in clustering
n_clusters = 12
msm_penalty = 1.0
msm_window = 0.5 # meaning 50% of the trajectory is considered for time warping

# Create the directory for the generated plots
output_dir = os.path.join(os.getcwd(), 'Clustering_results')
os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.join(os.getcwd(), f'Clustering_results/Agglomerative_tsMSM_separate_event_Clustering_cost{str(msm_penalty).replace(".", "p")}_window{str(msm_window).replace(".", "p")}')
os.makedirs(output_dir, exist_ok=True)
tsne_pca_output_dir = os.path.join(output_dir, "tSNE_and_PCA_analysis")
os.makedirs(tsne_pca_output_dir, exist_ok=True)

# Generate file to keep track of the residue pairs per cluster
res_pairs_file = os.path.join(output_dir, f'res_pairs_tsmsm_penalty{str(msm_penalty).replace(".", "p")}_window{str(msm_window).replace(".", "p")}.txt')
res_pairs_wrt = open(res_pairs_file, "w")

# Load topology file to get the residue sequence
PDB_top = md.load(topology_file, standard_names=False).topology
prot_indexes = PDB_top.select('protein')
sample_frame = md.load_frame(topology_file, 0, top=PDB_top)
n_residues = sample_frame.atom_slice(prot_indexes).n_residues

# Generate residue dictionaries
index_to_label_dict = {i: str(PDB_top.residue(i))[3:].zfill(len(str(n_residues - 1))) + str(PDB_top.residue(i))[:3]    for i in range(n_residues)}
label_to_index_dict = {i: res for res, i in index_to_label_dict.items()}
# If waters are sampled they are also included in the dictionaries
if waters_in_traj:
    index_to_label_dict[n_residues] = 'WATERS'
    label_to_index_dict['WATERS'] = n_residues
    n_residues += 1

# Load and sort replica directories 
q_analysis_file = os.path.join(f"{header}_Q_analysis.txt")
replica_directories = sorted([os.path.join('.', d) for d in os.listdir('.') if os.path.isdir(os.path.join('.', d)) and ('_unfolding_' in d or '_folding_' in d)],
    key=lambda x: ( int(re.search(r'event_(\d+)', x).group(1)), '_folding_' in x))
    # We sort base on event number and folding over unfolding events 
q_df = pd.read_csv(q_analysis_file, sep='\\s+', skiprows=1, header=None, names=['Frame', 'D.E.Shaw_Q', 'State',	'Traj_File', 'Frame_in_file'], low_memory=False)


# Variables to keep track of info for clustering
residue_pairs = [] # residue pairs to be clustered
collective_NCI_data = [] # NCI densities to base the clustering in 
q_values = [] # extracted Q values to compare with 

# Generate event names and calculate event lengths dynamically
event_labels, event_ticks, cumulative_length = [], [], 0

# We set the propeties or features used for clustering
density_expected_columns = [
    'SG Attractive n^1',    'SG VdW n^1',   'SG Repulsive n^1',
    #'SG Attractive n^4/3',  'SG VdW n^4/3', 'SG Repulsive n^4/3',
    #'SG Attractive n^5/3',  'SG VdW n^5/3', 'SG Repulsive n^5/3',
    #'SG Attractive n^2',    'SG VdW n^2',   'SG Repulsive n^2',
    #'SG Attractive n^2.5',  'SG VdW n^2.5', 'SG Repulsive n^2.5'
    ]

# Lastly we prepare a list for the distance array coming from each event
distance_matrices = []

# Collect data from the different replica directories 
for replica_dir in replica_directories:
    # from each directory we take data from the csv rediue pairs and the frame ranges from the relevant_data
    replica_dir_path = os.path.join(replica_dir, 'NCI_csvs')
    replica_relevant_data_file = os.path.join(replica_dir, 'relevant_data.txt')
    
    # Parse relevant_data.txt to know the sampled frames for each transition
    config.read(replica_relevant_data_file)
    start_frame = config.getint('Settings','Start Frame')
    end_frame = config.getint('Settings','End Frame')
    sampling_step = config.getint('Settings','Sampling step')
    sampled_frames = range(start_frame, end_frame+1 , sampling_step)

    # Extract every sampled frame corresponding Q value and add to the list one after the other
    cleaned_replica_dir = os.path.basename(replica_dir) 
    q_values += q_df[(q_df['Frame'].isin(sampled_frames))]['D.E.Shaw_Q'].to_list() #& (q_df['Traj_File'].str[:6] == cleaned_replica_dir[:6])

    # After extracting the Q_values from the sampled frames we also take the NCI densities from the csv files
    for filename in os.listdir(replica_dir_path):
        if '_densities' in filename and filename.endswith('.csv'):
            filepath = os.path.join(replica_dir_path, filename)
            pair_NCI_df = pd.read_csv(filepath)

            # From the filename we parse the involved residue pair
            if set(density_expected_columns).issubset(pair_NCI_df.columns):
                residue_pair = re.split(r'event_(\d+)_(\d+\w+)_vs_(\d+\w+)_densities.csv', filename)
                residue_index_pair = [label_to_index_dict[residue_pair[2]],label_to_index_dict[residue_pair[3]]] 

            # We will add up the data for each residue pair from the different folding/unfolding events
                # If it is the first time a residue is found, it is added to the residue_pair list
                event_pair_NCI_data = [pair_NCI_df[col].to_list() for col in density_expected_columns]
                for col_idx in range(len(density_expected_columns)):
                    std_dev = np.std(np.array(event_pair_NCI_data[col_idx]))
            # Normalize to have average 0 and std dev 1
                    if std_dev == 0.0:                 
                        event_pair_NCI_data[col_idx] = event_pair_NCI_data[col_idx] - np.mean(event_pair_NCI_data[col_idx])
                    else:
                        event_pair_NCI_data[col_idx] = (event_pair_NCI_data[col_idx] - np.mean(event_pair_NCI_data[col_idx])) / std_dev 

                if residue_index_pair not in residue_pairs:
                    residue_pairs.append(residue_index_pair)
                    collective_NCI_data.append(event_pair_NCI_data) #We take all smoothed and normalized output data to be used for clustering 
                else:
                    index = residue_pairs.index(residue_index_pair) 
                    for col_idx, col in enumerate(density_expected_columns): #If the residue pair has already been included in the series the new data is added on top
                        collective_NCI_data[index][col_idx] = np.concatenate((collective_NCI_data[index][col_idx], event_pair_NCI_data[col_idx]))
            else:
                print(f'Could not find expected density properties {density_expected_columns} in {filename}')
    
    # Get the number of time points in the event
    event_length = len(sampled_frames)
    print(replica_dir_path, ' event length:', event_length, 'NCI data length:', len(event_pair_NCI_data[0]))

    # Convert to time series and compute distance matrix, using exclusively the data from the last added event
    time_series_data = to_time_series_dataset(np.array(collective_NCI_data)[:, :, -event_length:])

    # Generate the distance matrix using MSM
    distance_matrix = msm_pairwise_distance(time_series_data, c=msm_penalty, window=msm_window)
    distance_matrices.append(distance_matrix)

    # We keep track of the length of each separate event for plotting
    cumulative_length += event_length
    event_ticks.append(cumulative_length) 

    # Generate event label (e.g., Folding 1, Unfolding 1, etc.)
    event_label = (replica_dir).split('/')[-1].replace(f'{header}_', '').replace('_', ' ')
    event_labels.append(" ".join(str(word) for word in event_label.split(' ')[1:]))
    print(event_label, ' : ', np.array(time_series_data).shape)

print('distance matrices',np.array(distance_matrix).shape)
mean_distance_matrix = np.mean(np.array(distance_matrices), axis=0)
time_points = np.arange(np.array(collective_NCI_data).shape[2])

# Set the colors for the clusters, we use tab10 for the main structural motifs 
tab10_colors = list(cm.tab10.colors) 

# Store results for elbow and silhouette methods
elbow_inertia, silhouette_scores, davies_bouldin_scores = [], [], []
cluster_labels_dict = {}


# Iterate over clustering values to both generate the clusters as well as indexes to check its validity
for n_cluster in range(2, n_clusters + 1):
    model = AgglomerativeClustering(n_clusters=n_cluster, metric='precomputed', linkage='complete')
    cluster_labels = model.fit_predict(mean_distance_matrix)

    # Sort clusters by the number of residue pairs they contain
    cluster_counts = np.bincount(cluster_labels)
    sorted_cluster_indices = np.argsort(cluster_counts)[::-1]  # Sort in descending order of population, so the larger and noisier groups are first
    # Update cluster_labels to reflect new ordering
    sorted_cluster_labels = np.zeros_like(cluster_labels)
    for new_idx, old_idx in enumerate(sorted_cluster_indices):
        sorted_cluster_labels[cluster_labels == old_idx] = new_idx + 1 # Add one to the cluster labels so that 0 is the empty diagonal 

    cluster_matrix = np.zeros((n_residues, n_residues))
    for i, pair in enumerate(residue_pairs): #Asign to each residue pair and cluster value its position in the matrix based on residue indexes
        res1, res2 = pair
        cluster_matrix[res1, res2] = sorted_cluster_labels[i]
        cluster_matrix[res2, res1] = sorted_cluster_labels[i]

    # Color the results using tab10 for structural motifs, white for the diagonal and 'lightgray', 'silver' for other less relevant contacts
    colors_needed = ['white', 'lightgray', 'silver'] + tab10_colors * ((n_cluster // 10) + 1)
    my_cmap = ListedColormap(colors_needed[:n_cluster + 1]) 

    # Plot cluster matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cluster_matrix, square=True, cmap=my_cmap, cbar=False, vmin=0, vmax=n_cluster, annot=False, ax=ax)
    # X-axis: Show only odd-numbered residue labels
    x_labels = [index_to_label_dict[j] if j % 2 == 0 else "" for j in range(n_residues)]
    ax.set_xticklabels(x_labels, rotation=90, fontsize=25)
    # Y-axis: Show only even-numbered residue labels
    y_labels = [index_to_label_dict[j] if j % 2 != 0 else "" for j in range(n_residues)]
    ax.set_yticklabels(y_labels, rotation=0, fontsize=25)
    ax.set_title('Agglomerative pairwise MSM dist clustering', fontsize=16)

    # Add legend for clusters
    legend_labels = [f"Cluster {i} ({len(np.where(sorted_cluster_labels == i)[0])})" for i in range(1, n_cluster + 1)]
    legend_colors = [my_cmap(i) for i in range(1, n_clusters + 1)]
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
    ax.legend(handles, legend_labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Agglomerative_tsMSM_separate_event_clustering_n{n_cluster}_cost{str(msm_penalty).replace(".", "p")}_window{str(msm_window).replace(".", "p")}.png'), dpi=300)
    plt.close(fig)
    cluster_labels_dict[n_cluster] = sorted_cluster_labels

    # Then plot the data for each cluster with time and compare it to Q 
    time_series_output_dir = os.path.join(output_dir, f'TimeSeries_Plots_n{n_cluster}_agglomerative_ts_pairwise_separate_event')
    os.makedirs(time_series_output_dir, exist_ok=True)
    #We go cluster index by cluster index 
    for cluster_idx in range(1,n_cluster+1):
        # Isolate the data points for the current cluster
        cluster_data_indices = np.where(sorted_cluster_labels == cluster_idx)[0]
        cluster_data = np.array(collective_NCI_data)[cluster_data_indices]  # Shape: (n_pairs_in_cluster, time points, properties)

        # We utilize the cluster average for clearer plotting, specially in more populated clusters
        cluster_average = np.mean(cluster_data, axis=0)  # Shape: (time points, properties)

        # Create a figure for the cluster
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # First subplot: Individual time series and cluster average for densities
        ax1 = axes[0]
        if cluster_idx > 1:
            for series in cluster_data:
                ax1.plot(time_points, series[2, :], color='red',   linewidth=1.0, alpha=0.2)  # Repulsive
                ax1.plot(time_points, series[1, :], color='green', linewidth=1.0, alpha=0.2)  # VdW
                ax1.plot(time_points, series[0, :], color='blue',  linewidth=1.0, alpha=0.2)  # Attractive
        # Plot cluster average for each densy
        ax1.plot(time_points, cluster_average[2, :], color='red', linewidth=2.5, label='SG Repulsive n^1 (Avg)')
        ax1.plot(time_points, cluster_average[1, :], color='green', linewidth=2.5, label='SG VdW n^1 (Avg)')
        ax1.plot(time_points, cluster_average[0, :], color='blue', linewidth=2.5, label='SG Attractive n^1 (Avg)')

        # Plot settings for normalized densities
        ax1.set_title(f'Cluster {cluster_idx} - Time Series', fontsize=16)
        ax1.set_ylabel('Density Values (z-score)', fontsize=14)
        ax1.legend(fontsize=12, loc='lower right')
        ax1.set_ylim(-3, 4)  

        ax1.set_xticks(event_ticks)
        ax1.grid(axis='x', linestyle='--', alpha=0.5)

        # Second subplot: Q values
        ax2 = axes[1]
        ax2.plot(time_points, q_values, color='orange', linewidth=2.5, label='Q (native contacts)')

        # Plot settings for Q values
        ax2.set_ylabel('Q Values', fontsize=14)
        ax2.legend(fontsize=12, loc='upper right')
        ax2.set_ylim(0, 1)  # Adjust y-axis based on expected range of Q values (modify if needed)
        
        # Inlcude event ticks and labels in the x-axis
        ax2.set_xticks(event_ticks)
        ax2.set_xticklabels(event_labels, rotation=45, ha='right', fontsize=10)
        ax2.grid(axis='x', linestyle='--', alpha=0.5)
        ax2.set_xlim(0, len(time_points))

        # Save the plot
        plot_path = os.path.join(time_series_output_dir, f'Cluster_{cluster_idx}_TimeSeries_with_Q_n{n_clusters}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)


    # For future calculations we generate a file with the residue pairs indexes in each cluster to easily use the data in future operations
    res_pairs_wrt.write(f"Results for n={n_cluster }\n")
    for cluster_id in range(2,n_cluster+1):
        cluster_pairs = [residue_pairs[i] for i in range(len(sorted_cluster_labels)) if sorted_cluster_labels[i] == cluster_id]
        res_pairs_wrt.write(f"Cluster {cluster_id}: {cluster_pairs }\n")
    res_pairs_wrt.write("\n")  # Separate results for readability


    # Lastly I want to use the cluster labels to score the quality of the clustering before  reseting them
    # Compute "inertia" (sum of squared distances within clusters)
    inertia = 0
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) <= 1:
            continue  # no aporta nada al cálculo de inertia
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        inertia += np.sum(cluster_distances) / (2 * len(cluster_indices))
    elbow_inertia.append(inertia)
    # Compute Silhouette Score (higher is better)
    if n_clusters > 1:  # Silhouette is only valid for n > 1
        silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
        silhouette_scores.append(silhouette_avg)

    # Compute Davies Bouldin score (lowe is better)
    davies_bouldin_scores.append(davies_bouldin_score(distance_matrix, cluster_labels))

res_pairs_wrt.close()
n_clusters_range = range(2, n_clusters+1)
# Plot Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(n_clusters_range, elbow_inertia, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Within-Cluster Distances (Inertia)")
plt.title("Elbow Method for Optimal Clusters")
plt.xticks(n_clusters_range)
plt.grid()
plt.savefig(os.path.join(output_dir, f'MSM_cost{str(msm_penalty).replace(".", "p")}_window_{str(msm_window).replace(".", "p")}_elbow_method.png'), dpi=300, bbox_inches="tight")

# Plot Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(n_clusters_range, silhouette_scores, marker='s', linestyle='-', color='g')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal Clusters")
plt.xticks(n_clusters_range)
plt.grid()
plt.savefig(os.path.join(output_dir, f'MSM_cost{str(msm_penalty).replace(".", "p")}_window_{str(msm_window).replace(".", "p")}_silhouette_analysis.png'), dpi=300, bbox_inches="tight")

# Plot Davies-Bouldin index
plt.figure(figsize=(8, 4))
plt.plot(n_clusters_range, davies_bouldin_scores, marker='^', linestyle='-', color='brown')
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Index for Clustering Quality")
plt.xticks(n_clusters_range)
plt.grid()
plt.savefig(os.path.join(output_dir, f'MSM_cost{str(msm_penalty).replace(".", "p")}_window_{str(msm_window).replace(".", "p")}_davies_bouldin_analysis.png'), dpi=300, bbox_inches="tight")

# Compute PCA and tSNE
pca = PCA(n_components=2)
pca_proj = pca.fit_transform(distance_matrix)  # PCA works directly on distance matrix
tsne = TSNE(n_components=2, metric="precomputed", init="random", perplexity=5, random_state=42)
tsne_proj = tsne.fit_transform(distance_matrix)

# Loop over clustering results
for n_clusters in n_clusters_range:
    cluster_labels = cluster_labels_dict[n_clusters]

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
