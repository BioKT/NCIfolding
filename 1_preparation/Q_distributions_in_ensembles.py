import os
import mdtraj as md
import numpy as np
import configparser
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import Patch

# Set the arial font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def DESHAW_q(traj, native): # Calculate exact Q values for the trajectories of sampled frames 
    BETA_CONST = 50
    LAMBDA_CONST = 1.2
    NATIVE_CUTOFF = 1.0
    alpha_carbons = native.topology.select_atom_indices('alpha')
    Ca_pairs = np.array([ (i, j) for (i, j) in combinations(alpha_carbons, 2)
        if abs(native.topology.atom(i).residue.index - native.topology.atom(j).residue.index) > 6])
    
    Ca_pairs_distances = md.compute_distances(native[0], Ca_pairs)[0]
    native_contacts = Ca_pairs[Ca_pairs_distances <= NATIVE_CUTOFF]
    r = md.compute_distances(traj, native_contacts)
    r0 = md.compute_distances(native[0], native_contacts)
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q

ensemble_dirs = [os.path.join('.', d) for d in os.listdir('.') if os.path.isdir(d) and '_Ensemble_' in d]
state_labels = [d.split('_')[-1] for d in ensemble_dirs]

q_data = []
for ensemble_dir, label in zip(ensemble_dirs, state_labels):
    config = configparser.ConfigParser()
    config.read(os.path.join(ensemble_dir, 'relevant_data.txt'))
    
    traj_file = os.path.join(ensemble_dir, config.get('Settings', 'trajectory_file'))
    top_file = os.path.join(ensemble_dir, config.get('Settings', 'topology_file'))
    native_file = os.path.join(ensemble_dir, config.get('Settings', 'native_state_file'))
    stride = config.getint('Settings', 'Sampling step')

    top = md.load(top_file, standard_names=False).topology
    traj = md.load(traj_file, top=top, stride=stride, standard_names=False)
    traj_ca = traj.atom_slice(top.select("name CA"))

    native = md.load_pdb(native_file)
    native_ca = native.atom_slice(native.topology.select("name CA"))

    q_values = DESHAW_q(traj_ca, native_ca)
    q_data.append((q_values, label))

# Style map for KDE plots
color_map = {'Folded': 'lightgray', 'Transition': 'orange', 'Unfolded': 'darkslategrey'}

# Plot histogram with broken axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4), gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.05)
# The histogram is plotted in the two parts of the graph
bins = np.linspace(0, 1, 50)
for q_values, label in q_data:
    weights = np.ones_like(q_values) / len(q_values)  # Normalize per series
    print(label, len(weights))
    ax1.hist(q_values, bins=bins, weights=weights, color=color_map[label], alpha=0.7)
    ax2.hist(q_values, bins=bins, weights=weights, color=color_map[label], alpha=0.7)

# Set limits and ticks for the graph
ax2.set_xlim(0, 1)
ax1.set_ylim(0.37, 0.433)
ax2.set_ylim(0, 0.19)
ax2.set_xticks(np.arange(0, 1.1, 0.1))
ax1.set_yticks(np.arange(0.39, 0.44, 0.03))
ax2.set_yticks(np.arange(0, 0.2, 0.03))
# Remove unnecesary axis elements
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(bottom=False, top=False, labelbottom=False)
ax2.xaxis.tick_bottom()

# Add diagonal lines (cut marks)in the graph cut
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# Add labels, the y one is included with fig.text so that is centered in the altogether y axis
ax2.set_xlabel("Q", fontsize=22)
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.text(-0.02, 0.55, 'Probability', va='center', rotation='vertical', fontsize=22)

# Define legend entries manually
legend_elements = [
    Patch(facecolor=color_map['Unfolded'], label='Unfolded'),
    Patch(facecolor=color_map['Transition'], label='Transition'),
    Patch(facecolor=color_map['Folded'], label='Folded')
    ]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=20, frameon=False, bbox_to_anchor=(0.5, 0.9))
ax2.tick_params(labelsize=20)
ax1.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("Q_density_broken_axis.png", dpi=400, bbox_inches='tight')