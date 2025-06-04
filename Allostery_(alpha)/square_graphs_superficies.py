import configparser
import os
import re
import pandas as pd
import mdtraj as md
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

# Set the arial font
plt.rcParams['font.family'] = 'Arial'

# Define the output directory for the graphs
output_directory = './Square_Graphs/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Within the working directory we find the NCI_csvs directories containing the compiled data, each corresponding to a different power of the interaction density (n)
csv_data_directories = [d for d in os.listdir('./') if re.match(r'NCI_csvs', d) and os.path.isdir(os.path.join('./', d))]

# Define custom color scales for the graphs, green for VdW and red to blue for attractive-repulsive
def create_custom_att_minus_rep_cmap():
    return LinearSegmentedColormap.from_list('red_blue', ['red', 'white', 'blue'], N=256)
def create_custom_vdw_cmap():
    return LinearSegmentedColormap.from_list('white_green', ['white', 'green'], N=256)

# This function rounds numbers to two significant figures (to be used on the ticks of the graph colorbar) 
def round_to_two_sigfigs(ticks):
    return [np.round(tick, 1 - int(np.floor(np.log10(abs(tick)))) if tick != 0 else 0) for tick in ticks]

# Extract settings from ./relevant_data.txt
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
try:
    header = config.get('Settings', 'header')
    trajectory_file = config.get('Settings', 'trajectory_file')
    topology_file = config.get('Settings', 'topology_file')
    chain1_residues = config.get('Settings', 'chain1')
    chain2_residues = config.get('Settings', 'chain2')
    start_frame = config.getint('Settings', 'Start Frame', fallback=0)
    end_frame = config.getint('Settings', 'End Frame', fallback=99) 
    sampling_step = config.getint('Settings', 'Sampling step', fallback=1)
    waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)
    case_study = config.get('Settings', 'case_study', fallback='split_transition')
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")


PDB_top = md.load(topology_file, standard_names=False).topology
traj = md.load(trajectory_file, top=PDB_top, standard_names=False)
n_residues = traj.n_residues

# Create a index-to-residue mapping and viceversa to label the corresponding ticks in the graph
residue_dict = {i: str(PDB_top.residue(i))[3:].zfill(len(str(n_residues-1))) + str(PDB_top.residue(i))[:3] for i in range(n_residues)}
index_dict = {residue_dict[i]: i for i in range(n_residues)}

if waters_in_traj == True:
    residue_dict[n_residues] = 'WATERS'
    index_dict['WATERS'] = n_residues
    n_residues += 1
# FIXED # n_residues = n_residues -1 #one residue is repeated for some reason 
# Based on the info from relevant_data.txt we know how many frames we can expect
if case_study == 'Folded' or case_study == 'Unfolded' or case_study == 'Unfolded' or case_study == 'Transition':
    n_of_output_images = 1
    studied_range = f'{case_study}_ensemble'
else:
    n_of_output_images = 10
    frames_per_image = (end_frame - start_frame +1) // n_of_output_images
    studied_range = f'{start_frame}_to_{end_frame}'

### Process each suffix directory is processed separately to yield its own set of results
csv_output_data = []
vdw_matrices = [np.zeros((n_residues, n_residues)) for _ in range(n_of_output_images)]
att_minus_rep_matrices = deepcopy(vdw_matrices)
# For that we iterate over the .csv files within the directory
for filename in os.listdir(r'./NCI_csvs'):
    if filename.endswith('.csv'):
        filepath = os.path.join('./NCI_csvs', filename)
        df = pd.read_csv(filepath)             

        # Now we parse the two interacting residues from the filename 
        residue_match = re.search(r'([^_]+)_vs_([^_]+)', filename)
        residue1 = residue_match.group(1)
        residue2 = residue_match.group(2)

        # Process each frame group 
        for i in range(n_of_output_images):                         # if studying the native or unfolded ensemble we wont separate by frame  
            first_frame_in_range = start_frame + i * frames_per_image if case_study == 'split_transition' else None
            last_frame_in_range = start_frame + (i + 1) * frames_per_image -1 if case_study == 'split_transition' else None
            # to create each graph we define a subset of the dataframe based 
            subset_df = df[(df['Frame'] >= first_frame_in_range) & (df['Frame'] <= last_frame_in_range)] if case_study == 'split_transition' else df

            # Fill matrices with SG data (for trajectory) or raw data (for native state)
            if case_study == 'Folded' or case_study == 'Unfolded' or case_study == 'Transition':
                vdw_matrices[i][index_dict[residue1], index_dict[residue2]] = subset_df['VdW n^1'].mean() 
                att_minus_rep_matrices[i][index_dict[residue1], index_dict[residue2]] = subset_df['Attractive n^1'].mean() - subset_df['Repulsive n^1'].mean() 
            else:
                vdw_matrices[i][index_dict[residue1], index_dict[residue2]] = subset_df['SG VdW n^1'].mean()
                att_minus_rep_matrices[i][index_dict[residue1], index_dict[residue2]] = subset_df['SG Attractive n^1'].mean() - subset_df['SG Repulsive n^1'].mean()

            # Append data to csv_output_data list
            csv_output_data.append({
                'Residue1': residue1,
                'Residue2': residue2,
                'Frame range': f'{first_frame_in_range} - {last_frame_in_range}',
                'vdW': vdw_matrices[i][index_dict[residue1], index_dict[residue2]],
                'Att - Rep': att_minus_rep_matrices[i][index_dict[residue1], index_dict[residue2]]
                })
                
# Calculate global max/min for color scaling
all_vals = np.concatenate([mat.flatten() for mat in att_minus_rep_matrices])
global_max_att_minus_rep = np.max(np.abs(all_vals))
global_max_vdw = max(np.max(mat) for mat in vdw_matrices)
    #global_max_att_minus_rep = 0.19
    #global_max_vdw = 0.49


# Convert to DataFrame and save as CSV
csv_output_df = pd.DataFrame(csv_output_data).sort_values(by=['Residue1', 'Residue2', 'Frame range'])
csv_output_filename = os.path.join(output_directory, f"{header}_Interactions_{studied_range}.csv")
csv_output_df.to_csv(csv_output_filename, index=False)
print(f"Generated CSV file: {csv_output_filename}")

chain1_contacts, chain2_contacts = [], []
# Parse ranges from chain1_residues and chain2_residues
chain1_start, chain1_end = int(chain1_residues.split()[0]), int(chain1_residues.split()[2])
chain2_start, chain2_end = int(chain2_residues.split()[0]), int(chain2_residues.split()[2])

# Create sets of residue labels belonging to each chain
chain1_labels = set([residue_dict[i] for i in range(chain1_start, chain1_end + 1)])
chain2_labels = set([residue_dict[i] for i in range(chain2_start, chain2_end + 1)])

# Iterate over the labels and categorize them
for res_label in csv_output_df['Residue1'].to_list() + csv_output_df['Residue2'].to_list():
    if res_label in chain1_labels and res_label not in chain1_contacts:
        chain1_contacts.append(res_label)
    elif res_label in chain2_labels and res_label not in chain2_contacts:
        chain2_contacts.append(res_label)

chain1_contacts = sorted(chain1_contacts) + [residue_dict[n_residues-2], residue_dict[n_residues-1]]
chain2_contacts = sorted(chain2_contacts) + [residue_dict[n_residues-2], residue_dict[n_residues-1]]


def plot_asymmetric_heatmap(matrix, x_labels, y_labels, title, cmap, vmin=None, vmax=None, colorbar_label=None, output_path=None):
    plt.figure(figsize=(len(x_labels) * 0.2 + 4, len(y_labels) * 0.2 + 4))  # Dynamic size based on label count
    ax = sns.heatmap(matrix, cmap=cmap, xticklabels=x_labels, yticklabels=y_labels, vmin=vmin, vmax=vmax, cbar_kws={'label': colorbar_label})

    ax.set_xlabel('Chain 2 Residues')
    ax.set_ylabel('Chain 1 Residues')
    ax.set_title(title)

    ax.vlines(x=len(chain2_contacts)-2, ymin=0, ymax=len(chain1_contacts), color="black", linewidth=0.2)
    ax.vlines(x=len(chain2_contacts)-1, ymin=0, ymax=len(chain1_contacts), color="black", linewidth=0.2)
    ax.hlines(y=len(chain1_contacts)-2, xmin=0, xmax=len(chain2_contacts), color="black", linewidth=0.2)
    ax.hlines(y=len(chain1_contacts)-1, xmin=0, xmax=len(chain2_contacts), color="black", linewidth=0.2)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"Saved: {output_path}")

# Plot individual frame-range heatmaps
for i in range(n_of_output_images):
    frame_range = f"{start_frame + i * frames_per_image}_to_{start_frame + (i + 1) * frames_per_image - 1}" if case_study == 'split_transition' else case_study

    output_vdw_path = os.path.join(output_directory, f"{header}_vdW_{frame_range}.png")
    output_attrep_path = os.path.join(output_directory, f"{header}_Att_minus_Rep_{frame_range}.png")

    # Filter contact matrix for current frame range
    vdw_matrix = np.zeros((len(chain1_contacts), len(chain2_contacts)))
    att_rep_matrix = np.zeros_like(vdw_matrix)

    for res1 in chain1_contacts:
        for res2 in chain2_contacts:
            if res1 in index_dict and res2 in index_dict:
                i1, i2 = index_dict[res1], index_dict[res2]
                vdw_matrix[chain1_contacts.index(res1), chain2_contacts.index(res2)] = vdw_matrices[i][i1, i2]
                att_rep_matrix[chain1_contacts.index(res1), chain2_contacts.index(res2)] = att_minus_rep_matrices[i][i1, i2]

    plot_asymmetric_heatmap(
        vdw_matrix,
        x_labels=chain2_contacts,
        y_labels=chain1_contacts,
        title=f'vdW Interactions ({frame_range})',
        cmap=create_custom_vdw_cmap(),
        vmin=0,
        vmax=global_max_vdw,
        colorbar_label='vdW Density',
        output_path=output_vdw_path
    )

    plot_asymmetric_heatmap(
        att_rep_matrix,
        x_labels=chain2_contacts,
        y_labels=chain1_contacts,
        title=f'Attractive - Repulsive Interactions ({frame_range})',
        cmap=create_custom_att_minus_rep_cmap(),
        vmin=-global_max_att_minus_rep,
        vmax=global_max_att_minus_rep,
        colorbar_label='Attractive - Repulsive Density',
        output_path=output_attrep_path
    )