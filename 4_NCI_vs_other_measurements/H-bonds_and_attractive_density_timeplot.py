import os
import re
import configparser
import mdtraj as md
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Set the arial font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Define output directory
output_directory = './H_bond_vs_attractive_dens/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Extract settings
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
header = config.get('Settings', 'header')

# Variables to keep track of info for correlation
trajs_total_hbonds, trajs_total_NCI_att_data = np.array([]), np.array([]) 
# Generate event names and calculate event lengths dynamically
event_labels, event_ticks, cumulative_frames = [], [], 0

# Find an sort transition directories
replica_directories = sorted([os.path.join('.', d) for d in os.listdir('.') if os.path.isdir(os.path.join('.', d)) and ('_unfolding_' in d or '_folding_' in d)],
    key=lambda x: ( int(re.search(r'event_(\d+)', x).group(1)), '_folding_' in x))

# Collect data from the different replica directories 
for replica_dir in replica_directories:
    # from each directory we take data from the csv rediue pairs and the frame ranges from the relevant_data
    replica_relevant_data_file = os.path.join(replica_dir, 'relevant_data.txt')
    
    # Parse relevant_data.txt to know the sampled frames for each transition
    config.read(replica_relevant_data_file)
    start_frame = config.getint('Settings','Start Frame')
    end_frame = config.getint('Settings','End Frame')
    sampling_step = config.getint('Settings','Sampling step')
    topology_file = config.get('Settings', 'topology_file')
    trajectory_file = config.get('Settings', 'trajectory_file')

    # Calculate the number of time points in the event
    sampled_frames = range(start_frame, end_frame +1, sampling_step)
    n_of_sampled_flames = len(sampled_frames)
    cumulative_frames += n_of_sampled_flames
    event_ticks.append(cumulative_frames) 

    # Generate event label (e.g., Folding 1, Unfolding 1, etc.)
    event_label = (replica_dir).split('/')[-1].replace(f'{header}_', '').replace('_', ' ')
    event_labels.append(event_label)

    # Now we take the trajectory data, but loading exclusively the sampled frames to calculate the number of H-bonds in said frames
    PDB_top = md.load(os.path.join(replica_dir, topology_file), standard_names=False).topology
    event_traj = md.load(os.path.join(replica_dir, trajectory_file), top=PDB_top, stride=sampling_step, standard_names=False)
    print(replica_dir, n_of_sampled_flames, 'frames')

    # Compute hydrogen bonds for each frame
    hbond_counts = []
    for frame in range(n_of_sampled_flames):
        detected_hbonds = md.baker_hubbard(event_traj[frame], exclude_water=True, periodic=True, sidechain_only=False)
        hbond_counts.append(len(detected_hbonds)) # we just care about how many, not between which atoms

    # Smooth H-bonds the same way that the NCI integrals
    smoothed_hbonds = signal.savgol_filter(np.array(hbond_counts), 10, 0) #window of 10 frames with no polynomial fitting
    trajs_total_hbonds = np.concatenate((trajs_total_hbonds, smoothed_hbonds))

    # Directory containing density CSVs
    replica_csvs_path = os.path.join(replica_dir, 'NCI_csvs')
    if not os.path.exists(replica_csvs_path ):
        raise FileNotFoundError(f"Directory {replica_csvs_path } not found.")

    NCI_att_data = np.zeros(n_of_sampled_flames)
    # Iterate over all CSVs to sum the attractive densities
    for filename in os.listdir(replica_csvs_path ):
        if '_densities' in filename and filename.endswith('.csv'):
            csv_filepath = os.path.join(replica_csvs_path , filename)
            NCI_df = pd.read_csv(csv_filepath)
            NCI_att_data += NCI_df['SG Attractive n^1'].to_numpy()

    # Then we add the data with the all attactive densities per sampled frame to the total list
    trajs_total_NCI_att_data = np.concatenate((trajs_total_NCI_att_data, NCI_att_data))


# PLOTTING
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8)) #Two graphs vertically stacked
fig.suptitle('Transition Time Series Data', fontsize=16)

# Plot number of H-bonds
ax1.plot(trajs_total_hbonds, color='dimgray')
ax1.set_ylabel('Number of H-bonds', color='dimgray', fontsize=24)
ax1.yaxis.label.set_position((ax1.yaxis.label.get_position()[0], 0.55))  # move label up
ax1.grid(axis='x', linestyle='--', color='gray', alpha=0.7)  # Vertical grid lines
ax1.set_yticks(ax1.get_yticks()[::2]) 
ax1.tick_params(axis='y', labelsize=22)
ax1.set_ylim(10, 35)  

# Plot total NCI attractive density
ax2.plot(trajs_total_NCI_att_data, color='blue')
ax2.set_ylabel(r'Total $\int \rho_{\text{attractive}}$', color='blue', fontsize=24)
ax2.yaxis.label.set_position((ax2.yaxis.label.get_position()[0], 0.45))  # move label down
ax2.grid(axis='x', linestyle='--', color='gray', alpha=0.7)  # Vertical grid lines
ax2.set_yticks(ax2.get_yticks()[::2]) 
ax2.tick_params(axis='y', labelsize=22)
ax2.set_ylim(0.5, 3.5)  

# Remove legends
ax1.legend().remove()
ax2.legend().remove()
# Remove x-axis label and leave only the ticks 
ax2.set_xlabel('')  
ax2.set_xticks(event_ticks)

# Label the xticks using the transition event labels
event_labels_simplified = [" ".join([label.split()[1], str(int(label.split()[3]))]) for label in event_labels]
ax2.set_xticklabels(event_labels_simplified, rotation=60, ha='right', fontsize=22)

# Trim x-axis to only show available data
plt.xlim(0, len(trajs_total_hbonds))
#plt.tight_layout()
fig.subplots_adjust(left=0.2, bottom=0.3)
plt.savefig(os.path.join(output_directory, 'time_series_plot_stacked.png'), dpi=300)