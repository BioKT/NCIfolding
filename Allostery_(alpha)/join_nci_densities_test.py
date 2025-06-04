import configparser
import re
import os
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

# VARIABLES FOR SAVITZKY-GOLAY SMOOTHING
moving_window_size = 10
fitting_polynomial_order = 0
plotting = False  # Set to False if you don't want to generate plots

# Define the grep patterns for the density powers of interest
grep_patterns = {
    'n=1.0': 'n^1',
    'n=4/3': 'n^4/3',
    'n=5/3': 'n^5/3',
    'n=2.0': 'n^2',
    'n=2.5': 'n^2.5'
    }

# Necessary data for file recognition is found in the relevant_data.txt file
config = configparser.ConfigParser()
config.read('./relevant_data.txt')
try:
    header = config.get('Settings', 'header')
    case_study = config.get('Settings', 'case_study', fallback='extracted_transition')
except configparser.NoOptionError as e:
    raise ValueError(f"Missing setting: {e}")
except ValueError as e:
    raise ValueError(f"Invalid value: {e}")

# Define the directories containing all the NCIplot data files and that for the csvs containing the compiled data
data_directory = './NCI_data/'
csv_directory = './NCI_csvs/'

# Create the output directory if it does not exist
if not os.path.exists(csv_directory):
    os.makedirs(csv_directory)

# We use a set to keep track of the processed NCIPLOT outputs
already_reviewed_files = set()

# Function to extract the densities of interest from the file
def extract_values(filepath, grep_pattern):
    dens_values = []
    with open(filepath, 'r') as file:
        count = 0
        for line in file:
            if grep_pattern.search(line):
                count += 1
                if count in [4, 6, 8]:
                    columns = line.split()
                    if len(columns) >= 3:
                        dens_values.append(float(columns[2]))
                if count > 8:
                    break
    return dens_values if len(dens_values) == 3 else print(f'Error in file {filepath}')

# Find all NCIPLOT outputs in the data directory
all_files = [f for f in os.listdir(data_directory) if f.endswith('.out')]

# Iterate over all files to process them all on a residue by residue pair order
for filename in all_files:
    if filename in already_reviewed_files:
        continue

    # If the file has not been already reviewed, it corresponds to a new residue pair, which will be parsed out
    match = re.match(re.escape(header) + r".*?_fr(\d+)_([^_]+)_vs_([^_]+)\.out", filename)
    if not match:
        print(f'Unrecognized output file: {filename}')
        continue
    
    residue1 = match.group(2)
    residue2 = match.group(3)

    # Find all files corresponding to that residue pair, meaning the results for different frames
    matching_files = [f for f in all_files if re.search(f"_{residue1}_vs_{residue2}\\.out$", f)]

    # Initialize a list to collect all the data for the current residue pair
    data = []

    for matching_file in matching_files:
        frame_match = re.search(r"_fr(\d+)_", matching_file)
        frame = int(frame_match.group(1))
        # A dictionary to contain all info from this residue pair is initiated 
        row = {'Frame': frame}

        # Extract densities for each grep pattern
        filepath = os.path.join(data_directory, matching_file)
        for pattern_str, suffix in grep_patterns.items():
            grep_pattern = re.compile(re.escape(pattern_str))
            dens_values = extract_values(filepath, grep_pattern)

            if dens_values:
                row[f'Attractive {suffix}'] = dens_values[0]
                row[f'VdW {suffix}'] = dens_values[1]
                row[f'Repulsive {suffix}'] = dens_values[2]

        if len(row) > 1:  # Ensure valid data was added
            data.append(row)

        # Mark this file as reviewed
        already_reviewed_files.add(matching_file)

    # Once all frames for this residue pair has been compiled create a DataFrame from the collected data
    if data:
        df = pd.DataFrame(data)
        df.sort_values(by='Frame', inplace=True)

        # Apply Savitzky-Golay smoothing and add it as new columns 
        for col in df.columns:
            if col != 'Frame':
                df[f'SG {col}'] = signal.savgol_filter(df[col], moving_window_size, fitting_polynomial_order)

        # Save the DataFrame to a CSV file
        csv_filename = os.path.join(csv_directory, f"{header}_{residue1}_vs_{residue2}_densities.csv")
        df.to_csv(csv_filename, index=False)

    if plotting:
        # Plot the smoothed densities for this residue pair
        plt.figure(figsize=(8, 6))

        # Choose columns based on case_study
        if case_study == 'extracted_transition':
            attractive_col = 'SG Attractive n^1'
            vdw_col = 'SG VdW n^1'
            repulsive_col = 'SG Repulsive n^1'
        else:
            attractive_col = 'SG Attractive n^1'
            vdw_col = 'SG VdW n^1'
            repulsive_col = 'SG Repulsive n^1'

        # Plot the data with corresponding colors
        plt.plot(df['Frame'], df[attractive_col], label=r'$\int \rho$ Attractive', color='blue', linewidth=2)
        plt.plot(df['Frame'], df[vdw_col], label=r'$\int \rho$ Van der Waals', color='green', linewidth=2)
        plt.plot(df['Frame'], df[repulsive_col], label=r'$\int \rho$ Repulsive', color='red', linewidth=2)

        # Labels and title
        plt.xlabel("Frames", fontsize=14)
        plt.ylabel(r"Inter-residue $\int \rho$", fontsize=14)
        plt.title(f"Inter-residue Densities: {residue1} vs {residue2}", fontsize=16)

        # Legend
        plt.legend(fontsize=12)

        # Save the figure
        plot_filename = os.path.join(csv_directory, f"{header}_{residue1}_vs_{residue2}_densities.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()