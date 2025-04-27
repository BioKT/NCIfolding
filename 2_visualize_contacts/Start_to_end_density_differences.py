import configparser
import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set the arial font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Define the output directory for the graphs
output_directory = './Start_to_End_Difference_Graphs/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Within the working directory we find the NCI_csvs directory containing the NCI density integral data
csv_directory = 'NCI_csvs'

# Define custom color scales for the graphs
def create_custom_att_minus_rep_cmap():
    return LinearSegmentedColormap.from_list('red_white_blue', ['red', 'white', 'blue'], N=256)

def create_custom_vdw_cmap():
    return LinearSegmentedColormap.from_list('purple_white_green', ['purple', 'white', 'green'], N=256)

# This function rounds numbers to two significant figures for color bar ticks
def round_to_two_sigfigs(ticks):
    return [np.round(tick, 1 - int(np.floor(np.log10(abs(tick)))) if tick != 0 else 0) for tick in ticks]

# Extract settings from ./relevant_data.txt
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
try:
    header = config.get('Settings', 'header')
    topology_file = config.get('Settings', 'topology_file')
    start_frame = config.getint('Settings', 'Start Frame', fallback=0)
    end_frame = config.getint('Settings', 'End Frame', fallback=10000) 
    sampling_step = config.getint('Settings', 'Sampling step', fallback=1)
    waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)
    case_study = config.get('Settings', 'case_study', fallback='transition')
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")

csv_directory = [d for d in os.listdir('./') if re.match(r'NCI_csvs(_\w+)?', d) and os.path.isdir(os.path.join('./', d))]

# Parse the topology to extract residues
import mdtraj as md
PDB_top = md.load(topology_file, standard_names=False).topology
frame = md.load_frame(topology_file, 0, top=PDB_top, standard_names=False)
prot_indexes = PDB_top.select('protein')
n_residues = frame.atom_slice(prot_indexes).n_residues

# Create residue index mappings
index_to_res_dict = {i: str(PDB_top.residue(i))[3:].zfill(len(str(n_residues-1))) + str(PDB_top.residue(i))[:3] for i in range(n_residues)}
res_to_index_dict = {index_to_res_dict[i]: i for i in range(n_residues)}
if waters_in_traj == True:
    index_to_res_dict[n_residues] = 'WATERS'
    res_to_index_dict['WATERS'] = n_residues
    n_residues += 1
print(index_to_res_dict)

# Process each directory
for csv_directory in csv_directory:
    suffix = csv_directory.split('_')[-1] if '_' in csv_directory else ''

    # Initialize matrices to store differences
    diff_vdw_matrix = np.zeros((n_residues, n_residues))
    diff_att_minus_rep_matrix = np.zeros((n_residues, n_residues))

    # Iterate over all .csv files in the directory
    for filename in os.listdir(os.path.join('./', csv_directory)):
        if filename.endswith('.csv'):
            filepath = os.path.join('./', csv_directory, filename)
            df = pd.read_csv(filepath)

            # Parse residue pair from the filename
            residue_match = re.search(r'([^_]+)_vs_([^_]+)', filename)
            residue1 = residue_match.group(1)
            residue2 = residue_match.group(2)

            # Calculate differences between the last and first frame
            first_frame = df[df['Frame'] == df['Frame'].min()]
            last_frame = df[df['Frame'] == df['Frame'].max()]

            diff_vdw = last_frame['SG VdW n^1'].values[0] - first_frame['SG VdW n^1'].values[0]
            diff_att_minus_rep = last_frame['SG Attractive n^1'].values[0] - first_frame['SG Attractive n^1'].values[0] - last_frame['SG Repulsive n^1'].values[0] + first_frame['SG Repulsive n^1'].values[0]

            diff_vdw_matrix[res_to_index_dict[residue1], res_to_index_dict[residue2]] = diff_vdw
            diff_att_minus_rep_matrix[res_to_index_dict[residue1], res_to_index_dict[residue2]] = diff_att_minus_rep

    # Calculate global max/min for scaling
    global_max_diff_att_minus_rep = max(abs(np.min(diff_att_minus_rep_matrix)), np.max(diff_att_minus_rep_matrix))
    global_max_diff_vdw = max(abs(np.min(diff_vdw_matrix)), np.max(diff_vdw_matrix))

    # Create combined difference matrix
    combined_matrix = diff_att_minus_rep_matrix + diff_vdw_matrix.T

    # Plot differences
    fig, ax = plt.subplots(figsize=(12, 8))

    # Upper triangle: attractive-repulsive differences
    mask_upper = np.zeros_like(combined_matrix, dtype=bool)
    mask_upper[np.tril_indices_from(mask_upper, k=-1)] = True
    att_minus_rep_cmap = create_custom_att_minus_rep_cmap()
    att_rep_ticks = round_to_two_sigfigs(np.linspace(-global_max_diff_att_minus_rep, global_max_diff_att_minus_rep, 9))
    sns.heatmap(combined_matrix, mask=mask_upper, ax=ax, cmap=att_minus_rep_cmap, square=True,
                vmin=-global_max_diff_att_minus_rep, vmax=global_max_diff_att_minus_rep,
                cbar_kws=dict(label=r'Difference in inter-residue $\int \rho_{\text{attractive}}$ - $\int \rho_{\text{repulsive}}$', location='bottom', pad=-0.025, shrink=0.4))

    # Lower triangle: VdW differences
    mask_lower = np.zeros_like(combined_matrix, dtype=bool)
    mask_lower[np.triu_indices_from(mask_lower, k=1)] = True
    vdw_cmap = create_custom_vdw_cmap()
    vdw_ticks = round_to_two_sigfigs(np.linspace(-global_max_diff_vdw, global_max_diff_vdw, 9))
    sns.heatmap(combined_matrix, mask=mask_lower, ax=ax, cmap=vdw_cmap, square=True,
                vmin=-global_max_diff_vdw, vmax=global_max_diff_vdw,
                cbar_kws=dict(label=r'Difference in inter-residue $\int \rho_{\text{vdW}}', location='bottom', pad=0.1, shrink=0.4))

    ax.set_title(f'{header} Difference between last and first frame ({suffix})')
    ax.set_xticks(np.arange(0.5, n_residues + 0.5, 1))
    ax.set_yticks(np.arange(0.5, n_residues + 0.5, 1))
    ax.set_xticklabels([index_to_res_dict[j] for j in range(n_residues)], rotation=90, fontsize=10)
    ax.set_yticklabels([index_to_res_dict[j] for j in range(n_residues)], rotation=0, fontsize=10)

    # Save plot
    plot_filename = os.path.join(output_directory, f"{header}_{suffix}_Start_to_End_Difference_Matrix.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=600)
    plt.close()
    print(f"Generated PNG file: {plot_filename}")