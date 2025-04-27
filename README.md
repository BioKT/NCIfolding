This is a collection of simple python scripts to prepare, compile and analyze NCIPLOT density data. 
To run the NCIPLOT calculations you need to install it (https://www.lct.jussieu.fr/pagesperso/contrera/nciplot.html)
Keep in mind that the scripts that call it, refer to it as nciplot to run the calculations

The scripts rely on the existence of a relevant_data.txt file to find necessary data for their function
In the case of the Q_analysis.py that we use to find the transitions in the trajectory or extract_transitions.py and extract_ensembles.py that we use to extract excerpts of the trajectory to study they will look like this:

[Settings]
header = NTL9-2-protein
trajectory_directory = NTL9-2-protein
topology_file = ./NTL9-2-protein.pdb
native_state_file = ./NTL9_alphafold.pdb
waters_in_traj = False

They header is included in outputed files so they can be differenciated and easily asigned to the studied system
There are a large number of trajectories in the trajectory_directory to go over ana analyse. We require a topology to load them and a native state to compare them
The scripts are geared to find and load .dcd files, but this is easily modified

The relevant_data.txt file we will more often use is present in the working directory of a trajectory which we are analyzing through NCIPLOT calculations, and will look like this:

[Settings]
header = NTL9-0-protein_folding_event_1
topology_file = ./NTL9-0-protein_folding_event_1_frame.pdb
native_state_file = ../NTL9_alphafold.pdb
trajectory_file = ./NTL9-0-protein_folding_event_1_traj.xtc
Start Frame: 126673
End Frame: 133041
Sampling step: 53

They include the topology and trajectory files to be loaded, the trajectory will be sampled to generate the NCIPLOT inputs and will be used for multiple analysis
They also state how separated are the sampled frames and the start and end points of this excerpt to facilitate its contextualization in the overall trajectory (if necessary)
