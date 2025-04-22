# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import sigmoid_growth_model as sgm

# Ensure text in exported PDF figures is editable
mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Helvetica'  # Set font to Helvetica

# Initialize the random number generator
rng = np.random.default_rng()

# Define simulation parameters
s_max = 2          # Maximum growth rate
k_e = 15           # Parameter that determines the shape of the fitness function
k_m = 4            # Parameter that determines the shape of the fitness function
                   # k_e = 1 and k_m = 0 implies ecDNA copy number-independent fitness.
t = 1.0            # Simulation time
n = 100            # Number of cells to simulate
min_copies = 1     # Minimum ecDNA copies in the cell population
max_copies = 20    # Maximum ecDNA copies in the cell population
p_s = 0.5          # Seperate probability of ecDNA

# Create a DataFrame to store simulation results
df = pd.DataFrame({
    'k_0': [],       # Initial ecDNA copies
    'N': [],         # Cell number of the clone
    't': [],         # Simulation time
    'ec': [],        # ecDNA copies of each cell in the clone
    'mean_ec': []    # Mean ecDNA copies of the clone
})

# Simulate the model
# Randomly select `n` cells with initial ecDNA copies from a uniform distribution
selected_cells = rng.integers(min_copies, max_copies, n)

# Run the simualtion for each selected cell
for k_0 in selected_cells:
    cell_num, simulation_time, copies_num_list = sgm.fix_time_model(
        s_max, k_e, k_m, k_0, t, p_s)
    
    # Calculate the mean ecDNA copies in the clone
    mean_ec = np.average(a=np.arange(0, 501), weights=copies_num_list[0:501])
    
    # Store the simulation data
    for ec_num in np.repeat(np.arange(0, 501), copies_num_list[0:501]):
        new_line = pd.DataFrame(np.array(
            [[k_0, cell_num, simulation_time, ec_num, mean_ec]]
        ), columns=['k_0', 'N', 't', 'ec', 'mean_ec'])
        df = pd.concat([df, new_line])

# Label data by the number of cells
df['N_label'] = pd.cut(df['N'], bins=[0, 1, 5, 9, 15, 500], labels=[
    '1', '2-5', '6-9', '10-15', '15+'])
df.reset_index(drop=True, inplace=True)

# Visualization
fig = plt.figure(figsize=(3., 2.5))
gs = fig.add_gridspec(10, hspace=0, height_ratios=[
                      2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5])
axs = gs.subplots(sharex=True, sharey=False)

# Define bins and color palette
bins = np.linspace(0, 30, 31)
color_plate = ['#355070', '#6d597a', '#b56576', '#e56b6f', '#eaac8b']

# Plot histograms, KDEs, and boxplots for each label
for i, label in enumerate(['1', '2-5', '6-9', '10-15', '15+']):
    sns.histplot(data=df[df['N_label'] == label], x='ec', bins=bins, ax=axs[i * 2],
                stat='density', fill=False, color='#5e6472')
    sns.kdeplot(data=df[df['N_label'] == label], x='ec', ax=axs[i * 2], color=color_palette[i],
                fill=True, alpha=0.72, linewidth=1.5, bw_adjust=0.5)
    sns.boxplot(data=df[df['N_label'] == label], x='ec', ax=axs[i * 2 + 1], color=color_palette[i],
                fill=False, linewidth=1, width=0.6, fliersize=2)

# Adjust subplot appearance
for ax in axs:
    ax.label_outer()
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.xaxis.set_ticks_position('none')
    
# Show x-axis ticks for the last subplot
axs[9].spines['bottom'].set_visible(True)
axs[9].xaxis.set_ticks_position('bottom')

# Set x-axis limits and ticks
plt.xlim(-5, 30)
plt.xticks(np.arange(0, 30, 10), fontsize=7)

axs[0].set_yticks([0], ['1'], fontsize=7)
axs[2].set_yticks([0], ['2-5'], fontsize=7)
axs[4].set_yticks([0], ['6-9'], fontsize=7)
axs[6].set_yticks([0], ['10-15'], fontsize=7)
axs[8].set_yticks([0], ['15+'], fontsize=7)

plt.xlabel('ecDNA copies number', fontsize=7.5, labelpad=2.5)

# Add y-axis label for the entire figure
fig.text(-0.05, 0.5, 'Number of cells', va='center',
        rotation='vertical', fontsize=7.5)

figure_path = '/Users/tutu/Documents/Project_ecDNA/article/figures/'
figure_name = 'ecDNA distribution with parameters s_max={}, k_e={}, k_m={}.pdf'.format(
    s_max, k_e, k_m)
plt.savefig(figure_path + figure_name, bbox_inches='tight', dpi=300, transparent=True)
