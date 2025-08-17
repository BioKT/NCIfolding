import configparser
import os
import pandas as pd
import mdtraj as md

#NUMBER OF SNAPSHOT TO SAMPLE:
n_snapshots = 125
ensemble_list = ['Folded', 'Transition', 'Unfolded']

# Load configuration and file paths
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
try:
    header = config.get('Settings', 'header')
    trajectory_directory = config.get('Settings', 'trajectory_directory')
    topology_file = config.get('Settings', 'topology_file')
    waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)
    timestep = config.getfloat('Settings', 'timestep', fallback=0.002)
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")

# We use the Q_analysis data file to find a number of random Transition snapshots
Q_analysis_file = f"{header}_Q_analysis.txt"
df = pd.read_csv(Q_analysis_file, sep=r'\s+')

for ensemble in ensemble_list:
    ensemble_traj = None
    # We sample on the csv and reset the indexes so they go from 1 to the n of sampled frames
    frames_to_sample = df[df['State'] == ensemble]
    selected_frames = frames_to_sample.sample(n=n_snapshots).reset_index(drop=True) 

    # We define an output directory in which we will generate both the NCI_inputs and geometries, as a trajectory
    output_dir = f"{header}_Ensemble_{ensemble}"
    os.makedirs(output_dir, exist_ok=True)

    # Process each selected frame
    for _, row in selected_frames.iterrows():
        frame = row['Frame']
        traj_file = row['Traj_File']
        frame_in_file = row['Frame_in_file']
        try:
            sampled_frame = md.load_frame(traj_file, index=frame_in_file, top=topology_file)
            sampled_frame = sampled_frame.make_molecules_whole() # Each frame must have whole molecules, PBCs do not translate to NCIPLOT
        except FileNotFoundError:
            print(f"Warning: Trajectory file {traj_file} not found. Skipping frame {frame}.")
            continue

        # The ensemble_traj is formed by adding every treated frame, one by one
        if ensemble_traj == None:
            ensemble_traj = sampled_frame
        else:
            ensemble_traj = md.join((ensemble_traj, sampled_frame), check_topology=True)

    ensemble_traj.save_xtc(os.path.join(output_dir,f'{header}_ensemble_{ensemble}.xtc'))    #trajectory 
    sampled_frame.save(os.path.join(output_dir,f"{header}_fr{frame}.pdb"))                  #topologu

    print(f'    trajectory saved in: {output_dir}')
    print(ensemble_traj)


    # Lastly, we generate a relevant_data.txt file for the ensemble to facilitate its analysis
    relevant_data_file = os.path.join(output_dir, "relevant_data.txt")
    with open(relevant_data_file, "w") as data:
            data.write("[Settings]\n")
            data.write(f"header = {header}_ensemble_{ensemble}\n")
            data.write(f"case_study = {ensemble}\n")
            data.write(f"topology_file = {header}_fr{frame}.pdb\n")
            data.write(f"trajectory_file = {header}_ensemble_{ensemble}.xtc\n")
            data.write(f"Start Frame = 0\n")
            data.write(f"End Frame = {n_snapshots-1}\n")
            data.write(f"Sampling step= 1\n")
            data.write(f"waters_in_traj = {waters_in_traj}\n")
print('if correctly used on dcd files, residues will appear duplicated. Use pymol to transform to pdbs and change the relevant_data file accordingly')
