import configparser
import os
import re
import pandas as pd
import mdtraj as md
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set the arial font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Define the output directory for the graphs
output_directory = './Square_Graphs/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Within the working directory we find the NCI_csvs directory containing the NCI density integral data
csv_directory = 'NCI_csvs'

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
    start_frame = config.getint('Settings', 'Start Frame', fallback=0)
    end_frame = config.getint('Settings', 'End Frame', fallback=10000) 
    sampling_step = config.getint('Settings', 'Sampling step', fallback=1)
    waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)
    case_study = config.get('Settings', 'case_study', fallback='extracted_transition')
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")


# Load traj to prepare the residue dictionaries
PDB_top = md.load(topology_file, standard_names=False).topology
traj = md.load(trajectory_file, top=PDB_top, standard_names=False)
prot_indexes = PDB_top.select('protein')
n_residues = traj.atom_slice(prot_indexes).n_residues

# Create a index-to-residue mapping and viceversa to label the corresponding ticks in the graph
residue_dict = {i: str(PDB_top.residue(i))[3:].zfill(len(str(n_residues-1))) + str(PDB_top.residue(i))[:3] for i in range(n_residues)}
index_dict = {residue_dict[i]: i for i in range(n_residues)}
if waters_in_traj == True:
    residue_dict[n_residues] = 'WATERS'
    index_dict['WATERS'] = n_residues
    n_residues += 1

# Based on the info from relevant_data.txt we know how many frames we can expect
if case_study == 'Folded' or case_study == 'Unfolded' or case_study == 'Transition':
    n_of_output_images = 1      #In the case of ensembles we take the total average, one image
    studied_range = f'{case_study}_ensemble'
else:
    n_of_output_images = 12 # for specific transitions we set a number of images to show its progress
    frames_per_image = (end_frame - start_frame) // n_of_output_images
    studied_range = f'{start_frame}_to_{end_frame}'


# Prepare a list to store data from the CSV output and set the matrices to be filled 
csv_output_data = []
vdw_matrices = [np.zeros((n_residues, n_residues)) for _ in range(n_of_output_images)]
att_minus_rep_matrices = [np.zeros((n_residues, n_residues)) for _ in range(n_of_output_images)]
# For that we iterate over the .csv files within the directory
for filename in os.listdir(os.path.join('./', csv_directory)):
    if filename.endswith('.csv'):
        filepath = os.path.join('./', csv_directory, filename)
        df = pd.read_csv(filepath)             

        # Now we parse the two interacting residues from the filename 
        residue_match = re.search(r'([^_]+)_vs_([^_]+)', filename)
        residue1 = residue_match.group(1)
        residue2 = residue_match.group(2)

        # Process each frame group 
        for i in range(n_of_output_images):                         # if studying the native or unfolded ensemble we wont separate by frame  
            first_frame_in_range = start_frame + i * frames_per_image if case_study == 'extracted_transition' else None
            last_frame_in_range = start_frame + (i + 1) * frames_per_image -1 if case_study == 'extracted_transition' else None
            # to create each graph we define a subset of the dataframe based 
            subset_df = df[(df['Frame'] >= first_frame_in_range) & (df['Frame'] <= last_frame_in_range)] if case_study == 'extracted_transition' else df

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
                    'VdW': vdw_matrices[i][index_dict[residue1], index_dict[residue2]],
                    'Att - Rep': att_minus_rep_matrices[i][index_dict[residue1], index_dict[residue2]]
                    })

# Convert to DataFrame and save as CSV
csv_output_df = pd.DataFrame(csv_output_data).sort_values(by=['Residue1', 'Residue2', 'Frame range'])
csv_output_filename = os.path.join(output_directory, f"{header}_Interactions_{studied_range}.csv")
csv_output_df.to_csv(csv_output_filename, index=False)
print(f"Generated CSV file: {csv_output_filename}")

# Calculate global max/min for color scaling
global_max_att_minus_rep = max( max(np.max(mat) for mat in att_minus_rep_matrices), min(np.min(mat) for mat in att_minus_rep_matrices))
global_max_vdw = max(np.max(mat) for mat in vdw_matrices)
#global_max_att_minus_rep = 0.19    #The limits we set for our case so they are the same throughout
#global_max_vdw = 0.49

# Plot and save matrices for each frame group
for i, (vdw_mat, att_rep_mat) in enumerate(zip(vdw_matrices, att_minus_rep_matrices)):
    combined_matrix = vdw_mat.T  + att_rep_mat #we fill the suqare with 2 triangles with the same positions and transposing one
    combined_df = pd.DataFrame(combined_matrix)

    fig, ax = plt.subplots(figsize=(11,10))
    fig.subplots_adjust(bottom=0.3, top=0.95) 

    # Upper triangle for attractive-repulsive interactions
    mask = np.zeros_like(combined_matrix, dtype=bool)
    mask[np.tril_indices_from(mask, k=-1)] = True
    att_minus_rep_cmap = create_custom_att_minus_rep_cmap()
    att_rep_ticks = round_to_two_sigfigs(np.linspace(-global_max_att_minus_rep, global_max_att_minus_rep, 5)) # 4 negative ticks, 0 and 4 positive ones  
    heatmap_upper = sns.heatmap(combined_df, mask=mask, ax=ax, cmap=att_minus_rep_cmap,  vmin=-global_max_att_minus_rep, vmax=global_max_att_minus_rep, 
                                    cbar=False, cbar_kws=dict(label=r'Inter-residue $\int \rho_{\text{attractive}}$ - $\int \rho_{\text{repulsive}}$' , location='bottom', pad=0.05, shrink=0.8))

    # Create a new axis for the  colorbar
    cbar_ax1 = fig.add_axes([0.2, 0.07, 0.6, 0.04])  #[left, bottom, width, height] — tuned for horizontal bar
    quadmesh_upper = [coll for coll in ax.collections if coll.get_cmap().name == 'red_blue'][0]
    cb1 = fig.colorbar(quadmesh_upper, cax=cbar_ax1, orientation='horizontal')
    cb1.set_label(r'Inter-residue $\int \rho_{\text{attractive}}$ - $\int \rho_{\text{repulsive}}$', fontsize=16)
    cb1.ax.tick_params(labelsize=14)


    # Lower triangle for VdW interactions
    mask = np.zeros_like(combined_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    vdw_cmap = create_custom_vdw_cmap()
    vdw_ticks = round_to_two_sigfigs(np.linspace(0, global_max_vdw, 5)) # the 0 and the same 4 positive ticks
    heatmap_lower = sns.heatmap(combined_df, mask=mask, ax=ax, cmap=vdw_cmap,   vmin=0, vmax=global_max_vdw,
                                    cbar=False, cbar_kws=dict(label=r'Inter-residue $\int \rho_{\text{vdW}}$', fontsize=16, location='bottom', pad=0.12, shrink=0.8))

    # Custom colorbar
    cbar_ax2 = fig.add_axes([0.2, 0.18, 0.6, 0.04])  # slightly lower
    quadmesh_lower = [coll for coll in ax.collections if coll.get_cmap().name == 'white_green'][0]
    cb2 = fig.colorbar(quadmesh_lower, cax=cbar_ax2, orientation='horizontal')
    cb2.set_label(r'Inter-residue $\int \rho_{\text{vdW}}$', fontsize=16)
    cb2.ax.tick_params(labelsize=14)


    # To set the title we check if we are studying any type of ensemble to name it accordingly 
    if case_study == 'Folded' or case_study == 'Unfolded' or case_study == 'Transition':
        ax.set_title(rf'Average $\int\rho$ for the {case_study} ensemble', fontsize=18)            
    else:    
        ax.set_title(rf'Average $\int\rho$ for frames {start_frame + i * frames_per_image} to {start_frame + (1+ i) * frames_per_image -1}', fontsize=18)

    ax.set_aspect('equal')
    ax.figure.axes[-1].xaxis.label.set_size(16)
    ax.figure.axes[-1].tick_params(axis='x', labelsize=14)        
    ax.set_xticks(np.arange(0.5, n_residues + 0.5, 1)) #I want ticks in the center of the respective square so that its clearer which residues are interacting
    ax.set_yticks(np.arange(0.5, n_residues + 0.5, 1))

    # X-axis: Show only odd-numbered residue labels
    x_labels = [residue_dict[j] if j % 2 == 0 else "" for j in range(n_residues)]
    ax.set_xticklabels(x_labels, rotation=90, fontsize=14)
    # Y-axis: Show only even-numbered residue labels
    y_labels = [residue_dict[j] if j % 2 != 0 else "" for j in range(n_residues)]
    ax.set_yticklabels(y_labels, rotation=0, fontsize=14)

    # Draw black square around the heatmap edges
    ax.hlines(y=0, xmin=0, xmax=n_residues, color="black", linewidth=2)  # Top border
    ax.hlines(y=n_residues, xmin=0, xmax=n_residues, color="black", linewidth=2)  # Bottom border
    ax.vlines(x=0, ymin=0, ymax=n_residues, color="black", linewidth=2)  # Left border
    ax.vlines(x=n_residues, ymin=0, ymax=n_residues, color="black", linewidth=2)  # Right border


    # Save plot as PNG
    if case_study == 'Folded' or case_study == 'Unfolded' or case_study == 'Transition':
        plot_filename = os.path.join(output_directory, f"{header}_Interaction_Matrix_of_{case_study}_ensemble.png")
    else:
        plot_filename = os.path.join(output_directory, f"{header}_NCI_Interaction_Density_Matrix_{start_frame + i * frames_per_image}_to_{start_frame + (1+ i) * frames_per_image -1}.png")

    plt.savefig(plot_filename,  dpi=600)
    plt.close()
    print(f"Generated PNG file: {plot_filename}")