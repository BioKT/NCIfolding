### NCI analysis of protein folding trajectories

This is a collection of  Python scripts to prepare, compile and analyze NCIPLOT density data. 
To run the NCIPLOT calculations, you first need to install it. You can download the software frm the [LCT webpage](https://www.lct.jussieu.fr/pagesperso/contrera/nciplot.html) or the [Github repository](https://github.com/juliacontrerasgarcia/NCIPLOT-4.0).
Keep in mind that the scripts that call this software refer to it as `nciplot` to run the calculations.

The scripts rely on the existence of a `relevant_data.txt` file to find necessary data for their function.
In the case of the `Q_analysis.py` that we use to find the transitions in the trajectory, or `extract_transitions.py` and `extract_ensembles.py` that we use to extract excerpts of the trajectory to study, they will look like this:

```
[Settings]
header = NTL9-2-protein
trajectory_directory = NTL9-2-protein
topology_file = ./NTL9-2-protein.pdb
native_state_file = ./NTL9_alphafold.pdb
waters_in_traj = False
```

The header is included in output files so they can be differentiated and easily assigned to the studied system.
There are a large number of trajectories in the `trajectory_directory` to go over and analyse. 
We require a topology file to load the trajectories and a native state file for reference.
The scripts are prepared to find and load `.dcd` files, but this can be easily modified.

The `relevant_data.txt` file we will more often use is present in the working directory of a trajectory, which we are analysing through NCIPLOT calculations, and will look like this:

```
[Settings]
header = NTL9-0-protein_folding_event_1
topology_file = ./NTL9-0-protein_folding_event_1_frame.pdb
native_state_file = ../NTL9_alphafold.pdb
trajectory_file = ./NTL9-0-protein_folding_event_1_traj.xtc
Start Frame: 126673
End Frame: 133041
Sampling step: 53
```

They include the topology and trajectory files to be loaded, the trajectory will be sampled to generate the NCIPLOT inputs and will be used for multiple analysis.
They also state how separated are the sampled frames and the start and end points of this excerpt to facilitate its contextualization in the overall trajectory (if necessary).


## Authors
* [Asier Urriolabeitia (@asieru25)](https://github.com/asieru25)
* [David De Sancho (@daviddesancho)](https://github.com/daviddesancho)
* [Julia Contreras García (@juliacontrerasgarcia)](https://github.com/juliacontrerasgarcia)
* Xabier López
