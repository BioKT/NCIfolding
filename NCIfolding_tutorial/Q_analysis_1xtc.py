import configparser
import os
import mdtraj as md
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

########################################################################################

# The necessary data to prepare the nci files is extracted from the file 'relevant_data.txt'
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
# Validate and extract the variables
try:
    header = config.get('Settings', 'header')
    trajectory_directory = config.get('Settings', 'trajectory_directory')
    topology_file = config.get('Settings', 'topology_file')
    native_state_file = config.get('Settings', 'native_state_file')
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")

#####################################################################################

# First we load a pdb to load the dcd trajectories which was generated using VMD on the DESHAW mae 
traj_top = md.load(topology_file).topology

# Additionally, we load the Alphafold pdb that we will use to calculate native contacts
native_state = md.load(native_state_file)
native_state_top = md.load(native_state_file).topology 

# IN THE CASE OF SHAW's TRAJECTORIES THEY ARE A SERIES OF DCD FILES WITH AN INDEX IN THEIR NAME
dcd_file_list = [] # We prepare the list of trajectories to be loaded and then analyzed
for root, dirs, files in os.walk(trajectory_directory):
    for file in files:
        if file.endswith('.xtc'):
            dcd_file_list = [os.path.join(root, file)]
dcd_file_list.sort() # order them based on the index in their name

# The data from consecutive dcd files is saved as a list 
data = []
frame_counter = 0 #frame counter will add the frames of the simulation we have already gone through
                  # so that we have a frame counter of the complete trajectory rather than each separate part

# We keep track of various data along to check if the transition path goes from Folded to Unfolded or vice versa
last_non_transition = 'Undetermined' #
transition_frames = [] # frames which are not clearly Folded or Unfolded (0.1 < Q < 0.9) are saved to be checked later 

def DESHAW_q(traj, native):
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.2  # from https://pubs.acs.org/doi/10.1021/jp110738m
    NATIVE_CUTOFF = 1.0  # nanometers

    alpha_carbons = native.topology.select_atom_indices('alpha')

    # Get residue pairs that are at least 7 residues apart
    Ca_pairs = np.array([
        (i, j) for (i, j) in combinations(alpha_carbons, 2)
        if abs(native.topology.atom(i).residue.index - 
               native.topology.atom(j).residue.index) > 6
    ])

    # Compute native distances directly
    native_distances = md.compute_distances(native[0], Ca_pairs)[0]
    # In some cases the computed distances seemed to produce artifacts, this alternative way to calculate it fixed it
    #native_distances = np.array([np.linalg.norm(native.xyz[0, pair[0], :] - native.xyz[0, pair[1], :]) for pair in Ca_pairs])
    
    # Select pairs where distance is ≤ 1.0 nm
    native_contacts = Ca_pairs[native_distances <= NATIVE_CUTOFF]
    #print("Number of native contacts:", len(native_contacts))

    # Compute distances for the full trajectory manually
    n_frames = traj.n_frames

    # Compute native distances (r) for the complete trajectory
    r = md.compute_distances(traj, native_contacts)
    # Compute native distances (r0)
    r0 = md.compute_distances(native[0], native_contacts)
    # Compute Q
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q


for dcd_file in dcd_file_list: 
    traj = md.load(dcd_file, top=traj_top)
    print(f'processing {dcd_file}') #   ; {traj.n_residues} residues')
    #traj = md.load(dcd_file, top=traj_top) #if using other file formats simply md.load can work
    
    traj_atom_indexes = traj_top.select("name CA") # for botht he trajectory and native state we exclusively use the CAs
    traj_data = traj.atom_slice(traj_atom_indexes)
    
    native_state_indexes = native_state_top.select("name CA")
    native_reference = native_state.atom_slice(native_state_indexes)
    
    # Compute Q values
    Q_results = DESHAW_q(traj_data, native_reference)

    for i, q in enumerate(Q_results):
        # Since trajectory fragments are loaded individually each start at frame 0
        frame = frame_counter + i # This way trajectories go one after another

        if q >= 0.9:
            state = 'Folded'
        # If there are any transition frames we come from a possible transition region and we must check if it is a valid transition path
            if len(transition_frames) > 0:
                # If it comes from a Folded region it was not a transition and all transition frames are reconverted to Folded
                if last_non_transition == 'Folded' or last_non_transition == 'Undetermined':
                    for actually_not_transtion_frame in transition_frames:
                        data[actually_not_transtion_frame][2] = 'Folded'
            
                # Otherwise it comes from an Unfolded structure and the Transition is valid
                elif last_non_transition == 'Unfolded':
                    print (f"Transition detected from frame {transition_frames[0]} to {transition_frames[-1]}")
                    
                last_non_transition = 'Folded' # now we are on a folded region
                # lastly transition frames are reset
                transition_frames = []

        elif q <= 0.1:
            state = 'Unfolded'
            if len(transition_frames) > 0:
                if last_non_transition == 'Unfolded' or last_non_transition == 'Undetermined':
                    for actually_not_transtion_frame in transition_frames:
                        data[actually_not_transtion_frame][2] = 'Unfolded'
                elif last_non_transition == 'Folded':
                    print (f"Transition detected from frame {transition_frames[0]} to {transition_frames[-1]}")
                    
                last_non_transition = 'Unfolded'                
                transition_frames = []
        
        elif last_non_transition == 'Undetermined':
            state = 'Undetermined'
            transition_frames.append(frame)

        else: # Any Q between 0.1 and 0.9 can be a transition
            state = 'Transition'
            transition_frames.append(frame) #thus is added to the transition frames to be recorded if apropiated


        data.append([frame, q, state, dcd_file, i]) # at the end of the loaded trajectory all of the data processed is logged into a list 
    frame_counter += len(Q_results) # Q_results = number of frames

# Check if the last recorded transition was incomplete
if transition_frames:
    print("Incomplete transition at the end, reverting to previous state.")
    for frame in transition_frames:
        data[frame][2] = last_non_transition

# Once all dcd files have been processed data is converted into a dataframe and saved as a csv 
df = pd.DataFrame(data, columns=['Frame', 'D.E.Shaw_Q', 'State', 'Traj_File', 'Frame_in_file'])
df.to_csv(f"{header}_Q_analysis.txt", sep='\t', index=False)


# Lastly we save a plot of the different transitions 
colors = {'Folded': 'lightgray', 'Unfolded': 'orange', 'Transition': 'darkslategray'}

# Create the scatter plot separating the dataframe based on the state value
plt.figure(figsize=(20, 5))
for state, color in colors.items():
    subset = df[df['State'] == state]
    plt.scatter(subset['Frame'], subset['D.E.Shaw_Q'], label=state, color=color, alpha=0.5)

# Set plot labels and title
plt.xlabel('Frame')
plt.ylabel('D.E.Shaw_Q')
plt.title(f'Frame vs native contacts as especified by D.E.Shaw for {header}')
plt.legend(title='State')
plt.grid(True)

# Save the plot to a file
plt.savefig(f'{header}_frame_vs_q_scatter_plot.png')