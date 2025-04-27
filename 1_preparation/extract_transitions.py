import configparser
import pandas as pd
import mdtraj as md
import os
import re

# Additional Frames at Start and End of Transition
extra_frames = 10
# Load necessary configuration data
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
# Extract the variables either to be used or to prepare the relevant_data.txt files for the produced transition trajectories
try:
    header = config.get('Settings', 'header')
    topology_file = config.get('Settings', 'topology_file')
    native_state_file = config.get('Settings', 'native_state_file')
    waters_in_traj = config.getboolean('Settings', 'waters_in_traj', fallback=False)
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")

# the dcd files are ordered based on their index, which we can use as integer to navigate among them
def get_file_index(dcd_file_name):
    match = re.search(r'(\d+)\.dcd', dcd_file_name)
    return int(match.group(1)) if match else None

# Load Q analysis file
Q_analysis_file = f"{header}_Q_analysis.txt"
try:
    df = pd.read_csv(Q_analysis_file, sep=r'\s+')
except FileNotFoundError:
    raise FileNotFoundError(f"The Q analysis file {Q_analysis_file} was not found")

# Initialize the variables to track transitions
transitions = []
start_frame, start_traj_file, start_frame_in_file = None, None, None
folding_count, unfolding_count = 0, 0
previous_state, in_transition = None, False

# Identify transition sequences from the Q analysis
for index, row in df.iterrows():
    frame, q, state, traj_file, frame_in_file = row['Frame'], row['D.E.Shaw_Q'], row['State'], row['Traj_File'], row['Frame_in_file']

    if state == 'Transition':
        if not in_transition:
            in_transition = True
            start_frame, start_traj_file, start_frame_in_file = frame, traj_file, frame_in_file
        if previous_state is None:
            previous_state = df.loc[index - 1, 'State'] if index > 0 else None

    else:
        if in_transition:
            end_frame = df.loc[index - 1, 'Frame']
            end_traj_file = df.loc[index - 1, 'Traj_File']
            end_frame_in_file = df.loc[index - 1, 'Frame_in_file']

            # A sampling set is determined so that about 100 evenly distributed frames represent the transition
            sampling_step = (end_frame - start_frame) // 100
            if sampling_step <= 0:
                sampling_step = 1

            # After that the start and end frame are redifined to include the extra frames before and after the transition
            # First we adjust the start frame
            try:
                if start_frame_in_file - extra_frames * sampling_step < 0:
                    # In this case we need frames from previous file
                    prev_file_index = get_file_index(start_traj_file) - 1
                    prev_file_name = start_traj_file.replace(f"{prev_file_index + 1:03}", f"{prev_file_index:03}")
                    previous_traj = md.load(prev_file_name, top=topology_file)
                    start_frame_in_file = previous_traj.n_frames + (start_frame_in_file - extra_frames * sampling_step)
                    start_traj_file = prev_file_name
                else:
                    start_frame_in_file -= extra_frames * sampling_step
                print(f'{start_traj_file} start frame: {start_frame_in_file}')
            except OSError:
                print(f"No previous file found for {start_traj_file}. Using start frame as-is.")
                start_frame_in_file = 0

            try:
                # Then we adjust the end frame 
                end_trajectory = md.load(end_traj_file, top=topology_file)
                if end_frame_in_file + extra_frames * sampling_step >= end_trajectory.n_frames:
                    # If we need frames from the next file
                    next_file_index = get_file_index(end_traj_file) + 1
                    next_file_name = end_traj_file.replace(f"{next_file_index - 1:03}", f"{next_file_index:03}")
                    end_frame_in_file = (end_frame_in_file + extra_frames * sampling_step) - end_trajectory.n_frames
                    end_traj_file = next_file_name
                else:
                    end_frame_in_file += extra_frames * sampling_step
                print(f'{end_traj_file} end frame: {end_frame_in_file}')
            except OSError:
                print(f"No next file found for {end_traj_file}. Using end frame as-is.")
                end_frame_in_file = end_trajectory.n_frames - 1

            # Determine the transition direction based on starting and ending states 
            if previous_state == 'Folded' and state == 'Unfolded':
                direction = 'unfolding'
                folding_count += 1
                count = folding_count
            elif previous_state == 'Unfolded' and state == 'Folded':
                direction = 'folding'
                unfolding_count += 1
                count = unfolding_count
            else:
                direction = 'unknown'

            output_dir = f"{header}_{direction}_event_{count}"

            # Log transition details
            transitions.append({
                'Output Directory': output_dir,
                'Start Frame': start_frame - extra_frames * sampling_step, #the frame was adjusted respective to the file, but not on the absolute number
                'End Frame': end_frame + extra_frames * sampling_step,      # so we adjust it now
                'First Trajectory File': start_traj_file,
                'Last Trajectory File': end_traj_file,
                'Start Frame in File': start_frame_in_file,
                'End Frame in File': end_frame_in_file,
                'Transition Direction': direction,
                'Sampling step': sampling_step
            })

            in_transition = False
            previous_state = state

# Process each transition
for transition in transitions:
    os.makedirs(transition['Output Directory'], exist_ok=True)
    output_path = os.path.join(transition['Output Directory'], f"{transition['Output Directory']}.xtc")
    start_traj_file = transition['First Trajectory File']
    start_idx = get_file_index(start_traj_file)
    start_frame_in_file = transition['Start Frame in File']
    end_idx = get_file_index(transition['Last Trajectory File'])
    end_frame_in_file = transition['End Frame in File']

    # we separatly load all the components on the transition to save it as only one in its respective directory 
    transition_traj = None
    for idx in range(start_idx, end_idx + 1): # in how many dcd files is the transition spread?
        traj_file_name = start_traj_file.replace(f"{start_idx:03}", f"{idx:03}")
        individual_traj = md.load(traj_file_name, top=topology_file)

        if idx == start_idx and idx == end_idx:
            traj_to_save = individual_traj[start_frame_in_file:end_frame_in_file + 1]
        elif idx == start_idx:
            traj_to_save = individual_traj[start_frame_in_file:]
        elif idx == end_idx:
            traj_to_save = individual_traj[:end_frame_in_file + 1]
        else:
            traj_to_save = individual_traj

        if transition_traj is None:
            transition_traj = traj_to_save
        else:
            transition_traj = md.join([transition_traj, traj_to_save])

    # save trajectory and topology for separate studies of each transition
    transition_traj.save_xtc(output_path)
    transition_traj[0].save_pdb(f"{output_path[:-4]}_frame0.pdb")

    # Generate a relevant_data.txt file for each transition so that we can work with it directly
    relevant_data_file = os.path.join(transition['Output Directory'], "relevant_data.txt")
    with open(relevant_data_file, "w") as data:
        data.write("[Settings]\n")
        data.write(f"header = {transition['Output Directory']}\n")
        data.write(f"topology_file = ./{transition['Output Directory']}_frame0.pdb\n")
        data.write(f"native_state_file = ../{native_state_file}\n")
        data.write(f"trajectory_file = ./{transition['Output Directory']}.xtc\n")
        data.write(f"Start Frame: {transition['Start Frame']}\n")
        data.write(f"End Frame: {transition['End Frame']}\n")
        data.write(f"Sampling step: {transition['Sampling step']}\n")
        if waters_in_traj == True:
            data.write(f"waters_in_traj: True\n")

# Additionally save relevant_data to be used by other scripts in this suite
with open(f"transition_pathways_for_{header}", "w") as log:
    for transition in transitions:
        log.write(f"Transition {transition['Output Directory']}:\n")
        log.write(f"    Start Frame: {transition['Start Frame']} ({transition['First Trajectory File']}, Frame in File: {transition['Start Frame in File']})\n")
        log.write(f"    End Frame: {transition['End Frame']} ({transition['Last Trajectory File']}, Frame in File: {transition['End Frame in File']})\n")
        log.write("\n")