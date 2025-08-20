import os
import re
import configparser
import mdtraj as md
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns  

# Set the arial font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Define output directory
output_directory = './H_bond_vs_attractive_dens/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# To extract settings for every trajectory
config = configparser.ConfigParser()

# Initialize arrays for each ensemble to include their corresponding data and set the number of data points per ensembles
ensemble_data = {'Folded': [], 'Unfolded': [], 'Transition': []}
ensemble_weights = {'Folded': 66, 'Transition': 1, 'Unfolded': 33}

# Variables to keep track of info for correlation
trajs_total_hbonds = np.array([])
trajs_total_NCI_att_data = np.array([])


# First we extract the data from the ensembles 
ensemble_dirs = [os.path.join('.', d) for d in os.listdir('.') if os.path.isdir(os.path.join('.', d)) and ('_Ensemble_' in d)]
state_labels = [dirname.split('_')[-1] for dirname in ensemble_dirs]

for ensemble_dir, state_label in zip(ensemble_dirs, state_labels):
    config.read(os.path.join(ensemble_dir, 'relevant_data.txt'))
    topology_file = config.get('Settings', 'topology_file')
    trajectory_file = config.get('Settings', 'trajectory_file')
    sampling_step = config.getint('Settings','Sampling step')

    # Load the data from the ensemble
    PDB_top = md.load(os.path.join(ensemble_dir, topology_file), standard_names=False).topology
    ensemble_traj = md.load(os.path.join(ensemble_dir, trajectory_file), top=PDB_top, stride=sampling_step, standard_names=False)

    # Calculate the number of H-bonds in each frame
    hbonds = np.array([len(md.baker_hubbard(frame, exclude_water=True, periodic=True, sidechain_only=False)) for frame in ensemble_traj])
    NCI_att_data = np.zeros(len(ensemble_traj))

    # Add up the attractive density in those frames 
    replica_csvs_path = os.path.join(ensemble_dir, 'NCI_csvs')
    for filename in os.listdir(replica_csvs_path):
        if '_densities' in filename and filename.endswith('.csv'):
            NCI_df = pd.read_csv(os.path.join(replica_csvs_path, filename))
            NCI_att_data += NCI_df['Attractive n^1'].to_numpy() # Since the scatter presents random separate points, 
                                                                #we do not want to smooth the data
    # Store the number of h bonds and total attractive density
    ensemble_data[state_label].extend(zip(hbonds, NCI_att_data))

print(f"Folded: {len(ensemble_data['Folded'])} -> {ensemble_weights['Folded']}; Unfolded {len(ensemble_data['Unfolded'])} -> {ensemble_weights['Unfolded']}; Transition {len(ensemble_data['Transition'])} -> {ensemble_weights['Transition']}")
for ensemble_dir, state_label in zip(ensemble_dirs, state_labels):
    # Convert list of tuples into a NumPy array
    ensemble_array = np.array(ensemble_data[state_label])

    # Sample indices without replacement
    sampled_indices = np.random.choice(len(ensemble_array), size=ensemble_weights[state_label], replace=False)

    # Select the sampled rows
    sampled_data = ensemble_array[sampled_indices]
    ensemble_data[state_label] = list(sampled_data)  # Replace with sampled data


# Compute global R²
all_hbonds, all_NCIs = zip(*sum(ensemble_data.values(), []))
all_hbonds = np.array(all_hbonds).reshape(-1, 1)
all_NCIs = np.array(all_NCIs)
model = LinearRegression().fit(all_hbonds, all_NCIs)
global_r2 = r2_score(all_NCIs, model.predict(all_hbonds))

# List of possible states
possible_states = ['Folded', 'Unfolded', 'Transition']
# I sort the data to be plot in that order so transitions are on top and more visible
available_states = [state for state in possible_states if state in ensemble_data]
ordered_states = [state for state in possible_states if state in available_states]

# Scatter & KDE plot
style_map = {'Folded': ('lightgray', 's'), 'Transition': ('orange', '*'), 'Unfolded': ('darkslategrey', '^')}
plt.figure(figsize=(8, 4.5))
for state in ordered_states:
    hbond_vals, NCI_vals = zip(*ensemble_data[state])
    color, marker = style_map[state]
    plt.scatter(hbond_vals, NCI_vals, color=color, label=f"{state} conformations", alpha=0.7, marker=marker)
    
sns.kdeplot(x=all_hbonds.flatten(), y=all_NCIs, levels=5, color='black', linewidths=1.5, label="Global KDE")
plt.xlabel('Number of H-bonds', fontsize=18)
plt.ylabel(r'Total $\int \rho_{\text{attractive}}$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16) 

legend_items = [plt.Line2D([0], [0], marker=style_map[state][1], color='w', markerfacecolor=style_map[state][0], markersize=15, label=f"{state} conformations") for state in style_map]
legend_items.append(plt.Line2D([0], [0], color='black', linewidth=1.5, label="Global KDE"))
plt.legend(handles=legend_items, loc='upper left', frameon=True, fontsize=18)
plt.text(0.85, 0.15, f"Global R² = {global_r2:.2f}", transform=plt.gca().transAxes, ha='right', fontsize=18, color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
plt.xlim(1, None)
plt.ylim(0.1, None)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'ensemble_data_scatter_plot.png'), dpi=400)