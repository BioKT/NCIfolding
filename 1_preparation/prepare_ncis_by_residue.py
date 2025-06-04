import configparser
import mdtraj as md
import numpy as np
import os
import pandas as pd
import re

# The necessary data to prepare the NCIplot inputs is taken form the file 'relevant_data.txt'
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
try:
    header = config.get('Settings', 'header')
    trajectory_file = config.get('Settings', 'trajectory_file')
    topology_file = config.get('Settings', 'topology_file')
    first_frame  = config.getint('Settings', 'Start Frame')
    last_frame  = config.getint('Settings', 'End Frame')
    sampling_step  = config.getint('Settings', 'Sampling step', fallback=1)
    waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")

#######################################################################################
def process_xyz_file(filepath):
    # Read the .xyz file into a pandas DataFrame, excluding the header 
    data = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=2)
    
    # Process the first column (atom label) to extract the first letter (exclusively taking the element) after ignoring any leading numbers
    data[0] = data[0].apply(lambda x: re.search(r'[A-Za-z]', x).group() if re.search(r'[A-Za-z]', x) else x)
    
    # Keep the header intact
    with open(filepath, 'r') as file:
        header = [next(file).strip() for _ in range(2)]  # Read the first two lines (header)
    
    # Put the header and then the data with the now atom symbols
    with open(filepath, 'w') as file:
        file.write('\n'.join(header) + '\n')
        data.to_csv(file, sep=' ', header=False, index=False)
########################################################################################

# First the trajectory and topology is loaded
PDB_top = md.load(topology_file, standard_names=False).topology
traj = md.load(trajectory_file, top = PDB_top, standard_names=False)

# We ensure all molecules are imaged whole because the NCIplot cannot take into account the PBCs to calculate the promolecular densities
prot_indexes = PDB_top.select('protein')
if waters_in_traj == True:
    # If we are interested in interactions between the protein and waters we first make all the molecules whole before centering
    traj.make_molecules_whole()
    prot_indixes = traj.topology.select('protein')

    # Compute the protein COM for every frame
    masses = np.array([atom.element.mass for atom in traj.topology.atoms])
    protein_masses = masses[prot_indixes]
    com = np.sum(traj.xyz[:, prot_indixes, :] * protein_masses[None, :, None], axis=1) / np.sum(protein_masses)

    # Compute box center in each frame (assuming orthorhombic box)
    box_center = 0.5 * traj.unitcell_lengths  # shape: (n_frames, 3)

    # Move everything so the protein is centered 
    traj.xyz = traj.xyz - com[:, None, :] + box_center[:, None, :]

    # Reimage all other molecules so they are placed near the protein
    traj.image_molecules(anchor_molecules='protein', inplace=True)
else:
    # If not, just making the molecule whole is enough 
    traj = traj.image_molecules(anchor_molecules=[[traj.topology.atom(i) for i in prot_indexes]] , make_whole = True, inplace=True) 


# We generate a dictionary for residue and water definitions to select them and extract their xyzs
n_residues = traj.atom_slice(prot_indexes).n_residues
definition_dict = {i: 'resid '+str(i)  for i in range(n_residues)}

# Additionally we generate a second dictionary in which the residue labels starts by the number to more easily order the generated files 
label_dict = {i: str(PDB_top.residue(i))[3:].zfill(len(str(n_residues-1))) + str(PDB_top.residue(i))[:3]   for i in range(n_residues)}
if waters_in_traj == True:
    definition_dict[n_residues] = 'waters'
    label_dict[n_residues] = 'WATERS'
print(f'The system includes: {label_dict}')

# We define the directory for the NCIPLOT inputs as well as the script to execute them 
output_dir = './NCI_data'
bash_script = 'make_ncis.sh'
n_xyz_files = 0
n_nciplot_inputs = 0

# Generate the input directory if not already created
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Header for the bash file to launch the NCIPLOT calculations 
bash_file = open(f"{output_dir}/{bash_script}", "w")
bash_file.write('#!/bin/bash \n')

# We go pair by pair generating the xyz files and nci inputs
for i in range(n_residues):
    atom_indexes_i = PDB_top.select(definition_dict[i])
    residue_traj_i = traj.atom_slice(atom_indexes_i)

    # Additionally we determine the water molecules around this residue throughout the selected trajectory snippet
    if waters_in_traj == True: 
        water_neighbors = md.compute_neighbors(traj, 1.0, atom_indexes_i, haystack_indices=PDB_top.select('water')) 

    # We generate every xyz for the first residue for each frame
    for frame in range(first_frame, last_frame + 1, sampling_step):
        frame_header = f"fr{str(frame).zfill(len(str(last_frame)))}"
        xyz_file_i = f"{header}_{frame_header}_{label_dict[i]}.xyz"
         
        # We have to use (frame-first_frame) because the clipped the trajectory starts at frame 0, but we want to keep track of the overall frame number
        residue_traj_i[(frame-first_frame)].save_xyz(f"{output_dir}/{xyz_file_i}")
        process_xyz_file(f"{output_dir}/{xyz_file_i}") # processing to go from atom types to simple atom symbols that NCIplot can recognize
        n_xyz_files += 1

        #Now we consider the interactions between this residue and others with higher residue index
        if i < n_residues - 2:
            for j in range(i+3, n_residues):  # Only include interactions between residues separated by at least 3 residues are considered          
                nci_header = f"{header}_{frame_header}_{label_dict[i]}_vs_{label_dict[j]}" 
                xyz_file_j = f"{header}_{frame_header}_{label_dict[j]}.xyz"
                with open(f"{output_dir}/{nci_header}.inp", "w") as nci_file:
                    nci_file.write('2 \n')
                    nci_file.write(str(xyz_file_i)+' \n')
                    nci_file.write(str(xyz_file_j)+' \n')
                    nci_file.write('OUTPUT 2 \n')
                    nci_file.write('COARSE \n')
                    nci_file.write('INCREMENTS 0.1 0.1 0.1 \n')
                    nci_file.write('CUTPLOT 0.05 0.5 \n')
                    nci_file.write('CUTOFFS 0.5 1.0 \n')
                    nci_file.write('INTERMOLECULAR \n')
                    nci_file.write('INTERCUT 0.85 0.75 \n')
                    nci_file.write('INTEGRATE \n')
                    nci_file.write(f"ONAME {nci_header} \n") #this is only included in case we want to change the type of output, the generated geometries are properly named
                    nci_file.write('RANGE 3 \n')
                    nci_file.write('-0.2 -0.02  \n')
                    nci_file.write('-0.02 0.02  \n') 
                    nci_file.write('0.02 0.2  \n') 
            
                bash_file.write(f"nciplot {nci_header}.inp > {nci_header}.out \n")
                n_nciplot_inputs += 1

    # Additionally, if the option is set, we consider the water molecules surrounding the residue in THIS FRAME
        if waters_in_traj == True: 
            water_xyz_file = f"{header}_{frame_header}_{label_dict[i][:3]}WATERS.xyz"
            water_frame = traj[(frame-first_frame)//sampling_step].atom_slice(water_neighbors[(frame-first_frame)//sampling_step])
            water_frame.save_xyz(f"{output_dir}/{water_xyz_file}")
            process_xyz_file(f"{output_dir}/{water_xyz_file}")       

            nci_header = f"{header}_{frame_header}_{label_dict[i]}_vs_WATERS"
            with open(f"{output_dir}/{nci_header}.inp", "w") as nci_file:
                nci_file.write('2 \n')
                nci_file.write(f"{xyz_file_i} \n")
                nci_file.write(f"{water_xyz_file}  \n")
                nci_file.write('OUTPUT 2 \n')
                nci_file.write('COARSE \n')
                nci_file.write('INCREMENTS 0.1 0.1 0.1 \n')
                nci_file.write('CUTPLOT 0.05 0.5 \n')
                nci_file.write('CUTOFFS 0.5 1.0 \n')
                nci_file.write('INTERMOLECULAR \n')
                nci_file.write('INTERCUT 0.85 0.75 \n')
                nci_file.write('INTEGRATE \n')
                nci_file.write(f"ONAME {nci_header} \n") 
                nci_file.write('RANGE 3 \n')
                nci_file.write('-0.2 -0.02  \n')
                nci_file.write('-0.02 0.02  \n') 
                nci_file.write('0.02 0.2  \n') 

            bash_file.write(f"nciplot {nci_header}.inp > {nci_header}.out \n")
            n_nciplot_inputs += 1

bash_file.close()
os.chmod(f"{output_dir}/{bash_script}", 0o777)
print(f"{n_xyz_files} xyz files created in {output_dir}")
print(f"{n_nciplot_inputs} NCIPLOT inputs created in {output_dir}")