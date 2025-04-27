import configparser
import os
import pandas as pd
import mdtraj as md
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set the arial font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Create output directory for storing results
output_directory = './Square_Graphs/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define color scales for the graphs
def create_green_cmap():
    return LinearSegmentedColormap.from_list('white_green', ['white', 'green'], N=256)
def create_blue_cmap():
    return LinearSegmentedColormap.from_list('white_blue', ['white', 'blue'], N=256)

# Load data from configuration file
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
try:
    header = config.get('Settings', 'header')
    trajectory_file = config.get('Settings', 'trajectory_file')
    topology_file = config.get('Settings', 'topology_file')
    start_frame = config.getint('Settings', 'Start Frame', fallback=0)
    end_frame = config.getint('Settings', 'End Frame', fallback=10000)
    case_study = config.get('Settings', 'case_study', fallback='split_transition')
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")

#######################################################################################################

# Load the trajectory
PDB_top = md.load(topology_file, standard_names=False).topology
traj = md.load(trajectory_file, top=PDB_top, standard_names=False)

# Create a dictionary to map residue indices to residue names
prot_indexes = PDB_top.select('protein')
n_residues = traj.atom_slice(prot_indexes).n_residues
residue_dict = {i: str(PDB_top.residue(i))[3:].zfill(len(str(n_residues - 1))) + str(PDB_top.residue(i))[:3] for i in range(n_residues)}
print(residue_dict)

# Adjust the number of output images depending on if we are studing an ensemble 
# or a specific transition in which we can see progress
if case_study == 'Folded' or case_study == 'Unfolded' or case_study == 'Transition':
    n_of_output_images = 1 # if dealing with an ensemble we want a unique image
else:
    n_of_output_images = 12
# set the frames per image, the sampling step used for NCI has no bearing in this, we want to use as much data as possible    
frames_per_image = traj.n_frames //  n_of_output_images

#######################################################################################################
# We use the mdtraj function to get the distances between heavy atoms, then set the -6th power for clearer plotting 
closest_atom_distances, residue_pairs = md.compute_contacts(traj, contacts='all', scheme='closest-heavy')
closest_inverse_distances = 1 / (closest_atom_distances**6)

# Find all hbonds, to keep track of them throughout the trajevtory
hbonds = md.baker_hubbard(traj, freq=0.0, exclude_water=True, periodic=True, sidechain_only=False)
all_Hbonds = []
inverse_hbond_distances = np.asarray([[[np.nan for _ in range(len(traj))] for _ in range(n_residues)] for _ in range(n_residues)])

# Set a dictionary of acceptors and donors to facilitate distance calculation between them 
for hbond in hbonds:
    donor_residue = PDB_top.atom(hbond[0]).residue
    acceptor_residue = PDB_top.atom(hbond[2]).residue
    donor_atom = PDB_top.atom(hbond[0])
    acceptor_atom = PDB_top.atom(hbond[2])
    all_Hbonds.append([donor_residue, acceptor_residue, donor_atom, acceptor_atom])

# Go hbond by hbond calculating the distances, even if the hbond is not always present 
hbonds_df = pd.DataFrame(all_Hbonds, columns= ['Donor Residue', 'Acceptor Residue','Donor Atom', 'Acceptor Atom'] )
for j in range(n_residues):
    for k in range(j, n_residues):
        if j < n_residues - 2: #We exclusively consider interactions between residues separated by at least 3 sequence positions
            hbond_distances = []
            residue_pair_df = hbonds_df[ ((hbonds_df['Donor Residue'] == PDB_top.residue(j)) & (hbonds_df['Acceptor Residue'] == PDB_top.residue(k))) 
                                        | ((hbonds_df['Donor Residue'] == PDB_top.residue(k)) & (hbonds_df['Acceptor Residue'] == PDB_top.residue(j))) ]
        
            for donor_atom , acceptor_atom in zip(residue_pair_df['Donor Atom'], residue_pair_df['Acceptor Atom']):
                hbond_distances.append((md.compute_distances(traj, [[donor_atom.index, acceptor_atom.index]])).flatten())

            if len(hbond_distances) > 0:
                inverse_hbond_distances[j][k] = 1 / np.min(hbond_distances, axis=0) # For each frame we take the closest distance between a pair of residues

############################################################################################################
avg_closest_inverse_distances = []
avg_inverse_hbond_distances = []

for i in range(traj.n_frames//frames_per_image):
    avg_closest_inverse_distances.append(np.mean(closest_inverse_distances[i*frames_per_image:(1+i)*frames_per_image], axis=0))
    avg_inverse_hbond_distances.append(np.mean(inverse_hbond_distances[:][:][i*frames_per_image:(1+i)*frames_per_image], axis=2))

contact_vmax = np.nanmax(np.array(avg_closest_inverse_distances))
hbond_vmax = np.nanmax(np.array(avg_inverse_hbond_distances))
#contact_vmax=1875  #used in our article 
#hbond_vmax=3.7     #used in our article 

for i in range(traj.n_frames//frames_per_image):
    # Create the contact matrices, using the mdtraj functionn squareform
    combined_matrix = np.mean(md.geometry.squareform(closest_inverse_distances[i*frames_per_image:(1+i)*frames_per_image], residue_pairs), axis=0)
  
    # Prepare diagonal matrix with hbond distances
    for j in range(n_residues):
        for k in range(j, n_residues):
            if abs(j - k) < 3:
                combined_matrix[j, k] = 0
                combined_matrix[k, j] = 0
            else:
                combined_matrix[j, k] = np.mean(inverse_hbond_distances[j][k][i*frames_per_image:(1+i)*frames_per_image], axis=0) 

    #PLOTING
    fig, ax = plt.subplots(figsize=(11,10))
    fig.subplots_adjust(bottom=0.3, top=0.95) 
    # Mask for upper-right triangle
    upper_right_mask = np.ones_like(combined_matrix, dtype=bool)
    upper_right_mask[np.triu_indices_from(upper_right_mask)] = False

    # Mask for lower-left triangle
    lower_left_mask = np.ones_like(combined_matrix, dtype=bool)
    lower_left_mask[np.tril_indices_from(lower_left_mask)] = False

    # Plot the upper-right: Alpha-carbon inverse distances
    heatmap_upper = sns.heatmap(combined_matrix, ax=ax, cmap=create_blue_cmap(), mask=upper_right_mask, square=True, 
                                vmin=0, vmax=hbond_vmax,  cbar=False) 
    ax.figure.axes[-1].xaxis.label.set_size(16)
    ax.figure.axes[-1].xaxis.set_label_coords(0.4, -1)
    ax.figure.axes[-1].tick_params(axis='x', labelsize=14)
    

    # Plot the lower-left: Closest-atom inverse distances
    heatmap_lower = sns.heatmap(combined_matrix, ax=ax, cmap=create_green_cmap(), mask=lower_left_mask, square=True, 
                                vmin=0, vmax=contact_vmax, cbar=False) 
    ax.figure.axes[-1].xaxis.label.set_size(16)
    ax.figure.axes[-1].xaxis.set_label_coords(0.4, -1)
    ax.figure.axes[-1].tick_params(axis='x', labelsize=14)

    # Create a new axis for the  colorbars
    cbar_ax1 = fig.add_axes([0.2, 0.07, 0.6, 0.04])  #[left, bottom, width, height] — tuned for horizontal bar
    quadmesh_upper = [coll for coll in ax.collections if coll.get_cmap().name == 'white_blue'][0]
    cb1 = fig.colorbar(quadmesh_upper, cax=cbar_ax1, orientation='horizontal')
    cb1.set_label(r'Inter-residue closest H donor' +'\n'+ r'and acceptor 1/r (nm$^{-1}$)', fontsize=16)
    cb1.ax.tick_params(labelsize=14)

    cbar_ax2 = fig.add_axes([0.2, 0.18, 0.6, 0.04])  # slightly lower
    quadmesh_lower = [coll for coll in ax.collections if coll.get_cmap().name == 'white_green'][0]
    cb2 = fig.colorbar(quadmesh_lower, cax=cbar_ax2, orientation='horizontal')
    cb2.set_label(r'Inter-residue closest heavy atom 1/r$^{6}$(nm$^{-6}$)', fontsize=16)
    cb2.ax.tick_params(labelsize=14)


    # Add title and axis labels
    if case_study == 'Folded':
        ax.set_title(f'Average inverse distances in the folded ensemble', fontsize=18)
    elif case_study == 'Transition':
        ax.set_title(f'Average inverse distances in the transition ensemble', fontsize=18)
    elif case_study == 'Unfolded':
        ax.set_title(f'Average inverse distances in the unfolded ensemble', fontsize=18)
    else:
        ax.set_title(f'Inverse distances for frames {start_frame + i*frames_per_image} to {start_frame + (i+1)*frames_per_image - 1}', fontsize=18)

    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, n_residues, 1))
    ax.set_yticks(np.arange(0.5, n_residues, 1))

    # X-axis: Show only odd-numbered residue labels
    x_labels = [residue_dict[j] if j % 2 == 0 else "" for j in range(n_residues)]
    ax.set_xticklabels(x_labels, rotation=90, fontsize=16)

    # Y-axis: Show only even-numbered residue labels
    y_labels = [residue_dict[j] if j % 2 != 0 else "" for j in range(n_residues)]
    ax.set_yticklabels(y_labels, rotation=0, fontsize=16)

    # Draw black square around the heatmap edges
    ax.hlines(y=0, xmin=0, xmax=n_residues, color="black", linewidth=2)  # Top border
    ax.hlines(y=n_residues, xmin=0, xmax=n_residues, color="black", linewidth=2)  # Bottom border
    ax.vlines(x=0, ymin=0, ymax=n_residues, color="black", linewidth=2)  # Left border
    ax.vlines(x=n_residues, ymin=0, ymax=n_residues, color="black", linewidth=2)  # Right border

    # Save the plot
    if case_study == 'Folded':
        plot_filename = os.path.join(output_directory, f"{header}_Inverse_Heavy_Atom_and_H_bond_Distances_Map_of_the_Folded_Ensemble_with_Cbars.png")
    elif case_study == 'Transition':
        plot_filename = os.path.join(output_directory, f"{header}_Inverse_Heavy_Atom_and_H_bond_Distances_Map_of_the_Transition_Ensemble_with_Cbars.png")
    elif case_study == 'Unfolded':
        plot_filename = os.path.join(output_directory, f"{header}_Inverse_Heavy_Atom_and_H_bond_Distances_Map_of_the_Unfolded_Ensemble_with_Cbars.png")
    else:
        plot_filename = os.path.join(output_directory, f"{header}_Inverse_Heavy_Atom_and_H_bond_Distances_Map_frames_{start_frame+i*frames_per_image}_to_{start_frame+(i+1)*frames_per_image - 1}_with_Cbars.png")

    plt.savefig(plot_filename, dpi=600)
    plt.close()
    print(f"Generated PNG for frames {start_frame+i*frames_per_image} to {start_frame+(1+i)*frames_per_image - 1}")