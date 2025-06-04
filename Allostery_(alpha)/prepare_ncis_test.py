import configparser
import mdtraj as md
import os
import pandas as pd
import re

# We define the directory to inputs as well as the script to execute them 
output_dir = './NCI_data'
bash_script = 'make_ncis.sh'
save_contacts = True 

# The necessary data to prepare the NCIplot inputs is taken form the file 'relevant_data.txt'
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
topology = md.load(topology_file, standard_names=False).topology
traj = md.load(trajectory_file, top = topology, standard_names=False)
# We ensure all molecules are imaged whole because the NCIplot cannot take into account the PBCs 
traj = traj.make_molecules_whole()
n_residues = traj.n_residues

definition_dict = {i: 'resid '+str(i) for i in range(n_residues)}
residue_dict = {i: str(topology.residue(i))[3:].zfill(len(str(n_residues-1))) + str(topology.residue(i))[:3]  for i in range(n_residues)}

# Prepare the indices for the different molecular entities
prot_atoms = topology.select('resid 0 to 453')  
chain1_atoms = topology.select('resid '+str(chain1_residues))
chain2_atoms = topology.select('resid '+str(chain2_residues))
lig_atoms = topology.select('resname LIG') # or resid 454
sub_atoms = topology.select('resname SUB') # or resid 455

# Store residue-residue pairs that are within 0.5 nm in any frame
protein_contacts = set()
ligand_contacts = set()
substrate_contacts = set()

# We first want to track the meaningful contacts to then prepare the NCIPLOT calculations 
for i in range(n_residues):
    group_i_atoms = topology.select(definition_dict[i])
    if i > n_residues-3:
        print(i, group_i_atoms)

    if group_i_atoms[0] in prot_atoms: 
        if group_i_atoms[0] in  chain1_atoms:
            neighbors = md.compute_neighbors(traj, 0.5, group_i_atoms, haystack_indices=chain2_atoms, periodic=True) 
        elif group_i_atoms[0] in  chain2_atoms:
            neighbors = md.compute_neighbors(traj, 0.5, group_i_atoms, haystack_indices=chain1_atoms, periodic=True)
        
        contacted_residues = set()
        for frame_neighbors in neighbors:
            for atom_index in frame_neighbors:
                res_j = topology.atom(atom_index).residue.index
                contacted_residues.add(res_j)

        res_i = topology.atom(group_i_atoms[0]).residue.index
        for res_j in contacted_residues:
            protein_contacts.add(tuple(sorted((res_i, res_j)))) # Use sorted tuple to avoid (A,B) and (B,A) duplicates


    elif group_i_atoms[0] in lig_atoms:
        neighbors = md.compute_neighbors(traj, 0.5, group_i_atoms, haystack_indices=prot_atoms, periodic=True)

        contacted_residues = set()
        for frame_neighbors in neighbors:
            for atom_index in frame_neighbors:
                res_j = topology.atom(atom_index).residue.index
                contacted_residues.add(res_j)

        ligand = topology.atom(group_i_atoms[0]).residue.index
        for res_j in contacted_residues:
            ligand_contacts.add(tuple(sorted((ligand, res_j))))


    elif group_i_atoms[0] in sub_atoms:
        neighbors = md.compute_neighbors(traj, 0.5, group_i_atoms, haystack_indices=prot_atoms, periodic=True)

        contacted_residues = set()
        for frame_neighbors in neighbors:
            for atom_index in frame_neighbors:
                res_j = topology.atom(atom_index).residue.index
                contacted_residues.add(res_j)

        substrate = topology.atom(group_i_atoms[0]).residue.index
        for res_j in contacted_residues:
            substrate_contacts.add(tuple(sorted((substrate, res_j))))

# For now ill just combine ombine all contacts
combined_contacts = set().union(*[protein_contacts, ligand_contacts, substrate_contacts])

# Create a set to store all residue indices involved in contacts
if save_contacts == True: 
    all_residues_in_contacts = set()
    for res_i, res_j in combined_contacts:
        all_residues_in_contacts.update([res_i, res_j])

    # Get all atom indices corresponding to those residues
    all_atoms_in_contacts = []
    for res_idx in all_residues_in_contacts:
        atom_indices = topology.select(definition_dict[res_idx])
        all_atoms_in_contacts.extend(atom_indices)

    # Remove duplicates and sort
    all_atoms_in_contacts = sorted(set(all_atoms_in_contacts))

    # Save to PDB
    traj.atom_slice(all_atoms_in_contacts).save(f"./{header}_residues_in_contact.pdb")
    print(f"Saved combined residues to ./{header}_residues_in_contact.pdb")


# To keep track of the generated files
n_xyz_files = 0
n_nciplot_inputs = 0

# After that a directory is created to put the outputs of this script
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# The script to launch the nciplots is prepared as the inputs are generated
bash_file = open(f"{output_dir}/{bash_script}", "w")
bash_file.write('#!/bin/bash \n')
written_xyz = set() # Track which xyz files have already been written

# We go pair by pair generating the xyz files and nci inputs
for res_i, res_j in combined_contacts:
    atom_indexes_i = topology.select(definition_dict[res_i])
    atom_indexes_j = topology.select(definition_dict[res_j])
    
    residue_traj_i = traj.atom_slice(atom_indexes_i)
    residue_traj_j = traj.atom_slice(atom_indexes_j)

    # We generate every xyz for the first residue for each frame
    for frame in range(start_frame, end_frame + 1, sampling_step):
        frame_header = f"fr{str(frame).zfill(len(str(end_frame)))}"
        xyz_file_i = f"{header}_{frame_header}_{residue_dict[res_i]}.xyz"
        xyz_file_j = f"{header}_{frame_header}_{residue_dict[res_j]}.xyz"
        nci_header = f"{header}_{frame_header}_{residue_dict[res_i]}_vs_{residue_dict[res_j]}"

        # Only create xyz for res_i if not done yet for this frame
        if xyz_file_i not in written_xyz:
            residue_traj_i[(frame-start_frame)].save_xyz(f"{output_dir}/{xyz_file_i}")
            process_xyz_file(f"{output_dir}/{xyz_file_i}")
            written_xyz.add(xyz_file_i)
            n_xyz_files += 1

        # Same for res_j
        if xyz_file_j not in written_xyz:
            residue_traj_j[(frame-start_frame)].save_xyz(f"{output_dir}/{xyz_file_j}")
            process_xyz_file(f"{output_dir}/{xyz_file_j}")
            written_xyz.add(xyz_file_j)
            n_xyz_files += 1

        #Now we consider the interactions between this residue and others with higher residue index
        with open(f"{output_dir}/{nci_header}.inp", "w") as nci_file:
                nci_file.write('2 \n')
                nci_file.write(str(xyz_file_i)+' \n')
                nci_file.write(str(xyz_file_j)+' \n')
                nci_file.write('OUTPUT 1 \n')
                nci_file.write('COARSE \n')
                nci_file.write('INCREMENTS 0.1 0.1 0.1 \n')
                nci_file.write('CUTPLOT 0.05 0.1 \n')
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

bash_file.close()
os.chmod(f"{output_dir}/{bash_script}", 0o777)
print(f"{n_xyz_files} xyz files created in {output_dir}")
print(f"{n_nciplot_inputs} NCIPLOT inputs created in {output_dir}")
