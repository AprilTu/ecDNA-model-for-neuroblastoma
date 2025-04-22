# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import sigmoid_growth_model as sgm


def plot_data(ax, df, color):
    for _, row in df.iterrows():
        ax.plot([0, 1], [row['cell_number_1'], row['cell_number_2']],
                '-', color='#DFDFDE', linewidth=0.8)
        ax.scatter([0, 1], [row['cell_number_1'], row['cell_number_2']],
                   color='#DFDFDE', s=10, zorder=0)
    ax.boxplot([df['cell_number_1'], df['cell_number_2']], positions=[0, 1], widths=0.35, sym='',
               showcaps=True, boxprops=dict(color=color, linewidth=1.5),
               medianprops=dict(color=color, linewidth=1.5), whiskerprops=dict(color=color, linewidth=1.5),
               capprops=dict(color=color, linewidth=1.5))


# Ensure text in exported PDF figures is editable
mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Helvetica'  # Set font to Helvetica

# Initialize the random number generator
rng = np.random.default_rng()

# Define simulation parameters
s_max = 2.0          # Maximum growth rate
k_e = 14           # Environmental impact parameter
k_m = 14           # Cell division rate
t_1 = 1.0          # Simulation time for clonogenic assay process
t_2 = 0.75         # Simulation time for treatment process
d_0, k_d, p = 1.5, 10, 0.5  # Toxicity parameters

# Create a DataFrame to store simulation results
df = pd.DataFrame(columns=[
    'k0',                # Initial ecDNA copies
    'cell_number_1',     # Cell number of the clone before treatment
    'cell_number_2',     # Cell number of the clone after treatment
    'mean_ec_1',         # Mean ecDNA copies of the clone before treatment
    'mean_ec_2',         # Mean ecDNA copies of the clone after treatment
    'total_ec_1',        # Total ecDNA of the clone before treatment
    'total_ec_2'         # Total ecDNA of the clone after treatment
])

# Initialize counters for different cell number ranges
n_1, n_2, n_3 = 0, 0, 0

# Simulate until each range has at least 50 samples
while n_1 < 50 or n_2 < 50 or n_3 < 50:
    k_0 = rng.integers(1, 40, 1)[0]  # Randomly select initial ecDNA copies

    # Clonogenic assay
    cell_num_1, simulation_time_1, copies_num_list_1 = sgm.fix_time_model(
        s_max, k_e, k_m, k_0, t_1
    )

    # Categorize based on cell number
    if 15 < cell_num_1 <= 40:
        n_3 += 1
    elif 2 <= cell_num_1 <= 5:
        n_2 += 1
    elif 6 <= cell_num_1 <= 9:
        n_1 += 1

    # Simulate treatment
    cell_num_2, simulation_time_2, copies_num_list_2 = sgm.treatment_model_b(
        copies_num_list_1, s_max, k_e, k_m, t_2, d_0, k_d, p
    )

    # Calculate ecDNA statistics
    ec_1 = np.repeat(np.arange(0, 100, 1), copies_num_list_1[0:100])
    ec_2 = np.repeat(np.arange(0, 100, 1), copies_num_list_2[0:100])

    mean_ec_1 = np.mean(ec_1)
    mean_ec_2 = np.mean(ec_2)

    total_ec_1 = np.sum(ec_1)
    total_ec_2 = np.sum(ec_2)

    # Append results to the DataFrame
    df = pd.concat([df, pd.DataFrame({
        'k0': k_0,
        'cell_number_1': cell_num_1,
        'cell_number_2': cell_num_2,
        'mean_ec_1': mean_ec_1,
        'mean_ec_2': mean_ec_2,
        'total_ec_1': total_ec_1,
        'total_ec_2': total_ec_2
    }, index=[0])], ignore_index=True)

# Fill missing values with 0
df.fillna(0, inplace=True)

# Visualization
# Select data for plotting
df_0 = df[(df['cell_number_1'] >= 2) & (df['cell_number_1'] <= 5) & (df['cell_number_2'] > 0)][:50]
df_1 = df[(df['cell_number_1'] >= 6) & (df['cell_number_1'] <= 9) & (df['cell_number_2'] > 0)][:50]
df_2 = df[(df['cell_number_1'] > 15) & (df['cell_number_1'] < 40) & (df['cell_number_2'] > 0)][:50]

# Plot cell number distributions
fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.3))
colors = ['#6d597a', '#b56576', '#eaac8b']
dataframes = [df_0, df_1, df_2]
linewidths = [0.7, 1, 1]

for i, (df_, color, lw) in enumerate(zip(dataframes, colors, linewidths)):
    plot_data(axs[i], df_, color, lw)
    axs[i].set_xlim([-0.5, 1.5])
    axs[i].set_ylim([0, 40])
    axs[i].set_xticks([0, 1], ['Untreated', 'Treatment'])
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)

axs[0].set_ylabel('Cell number')

figure_name = 'treatment cell number s_max={}, k_e={}, k_m={}.pdf'.format(
    s_max, k_e, k_m)
plt.savefig(figure_path + figure_name, bbox_inches='tight', dpi=300, transparent=True)










