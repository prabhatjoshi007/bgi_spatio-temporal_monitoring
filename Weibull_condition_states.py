#%%
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pyswmm import Simulation
import glob
from compute_k_lamda import compute_k_lamda

#%%
path_codes = r'C:\Users\joshipra\switchdrive\BGI_simulator\08_codes_and_functions'
os.chdir(path_codes)

from Ks_to_por import Ks_to_por

 

def weibull_pdf(q, lambda_j, k_j):
    """Weibull probability density function."""
    return (k_j / lambda_j) * (q / lambda_j) ** (k_j - 1) * np.exp(- (q / lambda_j) ** k_j)

def weibull_survival(q, lambda_j, k_j):
    """Weibull survival function."""
    return np.exp(- (q / lambda_j) ** k_j)


def compute_cumulative_pdfs(pdfs, q, delta):
    """Compute cumulative PDFs using convolution."""
    cumulative_pdfs = [pdfs[0]]
    cumulative_pdf = pdfs[0]  # Start with the first state's PDF
    for pdf in pdfs[1:]:
        # Convolve the current cumulative PDF with the next state's PDF
        cumulative_pdf = fftconvolve(cumulative_pdf, pdf, mode="full")[:len(q)] * delta
        cumulative_pdfs.append(cumulative_pdf)
    return cumulative_pdfs

def compute_cumulative_survivals(cumulative_pdfs, q, delta):
    """Compute cumulative survival curves from cumulative PDFs."""
    cumulative_survivals = []
    for cum_pdf in cumulative_pdfs:
        cdf = np.cumsum(cum_pdf) * delta  # Cumulative Distribution Function
        survival = 1 - cdf  # Survival Function
        cumulative_survivals.append(survival)
    return cumulative_survivals


def compute_transition_probability_matrices(num_states, num_years, lambdas, k):
    """
    Compute transition probability matrices for each time step based on Eq 12,
    ensuring P[j, j] does not decrease across consecutive timesteps.

    Parameters:
    - num_states: Number of condition states.
    - num_years: Total number of years to compute.
    - lambdas: List of scale parameters for each state.
    - k: List of shape parameters for each state.

    Returns:
    - transition_matrices: List of transition matrices for each time step.
    """
    # Time vector
    q = np.linspace(0, num_years, num_years * 365)  # Daily timesteps
    delta = q[1] - q[0]  # Time step size

    # Compute PDFs and survival functions for each state
    pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
    #surv = [weibull_survival(q, lambdas[j], k[j]) for j in range(num_states)]
    cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
    cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)

    # Initialize list to hold transition matrices
    transition_matrices = []
    #previous_P = None  # To store the previous transition matrix

    # Compute transition matrices for each time step
    for x in range(num_years * 365):
        P = np.zeros((num_states + 1, num_states + 1))  # Transition matrix

        for j in range(num_states):
            if j == 0:  # First state
                num = np.round(cumulative_pdfs[j][x], 6)
                den = np.round(cumulative_survivals[j][x], 6)
                if den > 0:
                    P[j, j + 1] = num * delta/ den if num * delta / den < 1 else 1
                    P[j, j] = 1 - P[j, j + 1]
                else:
                    P[j, j + 1] = 1
                    P[j, j] = 0
            else:  # Subsequent states
                num = cumulative_pdfs[j][x]
                den = cumulative_survivals[j][x] - cumulative_survivals[j - 1][x]
                if den > 0:
                    P[j, j + 1] = num * delta / den if num *delta / den < 1 else 1
                    P[j, j] = 1 - P[j, j + 1]
                else:
                    P[j, j + 1] = 1
                    P[j, j] = 0

        # Final absorbing state
        P[-1, -1] = 1
        
        
        '''
        # Ensure P[j, j] does not decrease compared to the previous step
        if previous_P is not None:
            for j in range(num_states):
                if P[j, j] > previous_P[j, j]:
                    P[j, j] = previous_P[j, j]
                    P[j, j + 1] = 1 - P[j, j]
                    
                    '''                    

        # Save the current transition matrix and update the previous
        transition_matrices.append(P)
        #previous_P = P.copy()

    return transition_matrices


def plot_cumulative_survivals(cumulative_survivals, q):
    """
    Plot cumulative survival functions for all condition states.

    Parameters:
    - cumulative_survivals: List of cumulative survival functions for all condition states.
    - q: Time vector (discretized time steps).
    """
    plt.figure(figsize=(8, 6))
    for idx, survival in enumerate(cumulative_survivals):
        plt.plot(q, survival, label=f'Survival {idx + 1}')
    plt.xlabel('Time [years]')
    plt.ylabel('Survival Probability')
    plt.title('Cumulative Survival Functions for Condition States')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_cumulative_pdfs(cumulative_pdfs, q):
    """
    Plot cumulative PDFs for all condition states.

    Parameters:
    - cumulative_pdfs: List of cumulative PDFs for all condition states.
    - q: Time vector (discretized time steps).
    """
    plt.figure(figsize=(8, 6))
    for idx, cum_pdf in enumerate(cumulative_pdfs):
        plt.plot(q, cum_pdf, label=f'Cumulative PDF {idx + 1}')
    plt.xlabel('Time [years]')
    plt.ylabel('Probability Density')
    plt.title('Cumulative PDFs for Condition States')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
k, lambdas = compute_k_lamda(num_states = 4)


#lambdas = [4.6829, 4.1966, 4.8617, 4.9165]  # Scale parameters for condition states
lambdas = [4.987, 7.929, 4.384, 2.264]  # Scale parameters for condition states

k = [3.568, 6.585, 4.00, 2.961]                     # Shape parameter for Weibull distribution
delta = 1/365              # Time step (in d)
num_years = 25
num_days = 365 * num_years         # 25 years of daily transitions
num_states = 4
q =  np.linspace(0, num_years, num_days)


# Compute the transition matrices for each day ; (lambdas, k, delta, max_age, num_days)
transition_matrices = compute_transition_probability_matrices(num_states, num_years, lambdas, k)

pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)

# Plot survival functions

plot_cumulative_survivals(cumulative_survivals, q)

plot_cumulative_pdfs(cumulative_pdfs, q)


# Display the transition matrix for the first day
print("Transition Probability Matrix for Day 1:")
print(transition_matrices[0])

# Display the transition matrix for the last day
print("Transition Probability Matrix for Day", num_days, ":")
print(transition_matrices[-1])


# Initialize the state vector
state_vector = np.array([1, 0, 0, 0, 0])  # Initial state



# Transition matrices (list of matrices for each timestep)
# Assume `transition_matrices` is already computed and contains matrices for each timestep
# transition_matrices[i] is the transition matrix at timestep i

# Storage for state vectors over time
state_vectors = np.zeros((num_days, len(state_vector)))
state_vectors[0] = state_vector

# Iterate through each timestep
for t in range(1, num_days):
    #print("Old state vector:", state_vector)
    state_vector = np.dot(state_vector, transition_matrices[t])
    #print("New state vector:", state_vector)
    state_vectors[t] = state_vector
    
    
# Step 1: Determine the most likely state for each day
most_likely_states = np.argmax(state_vectors, axis=1)  # Get column index of max probability

# Output the results
print("Most Likely States for Each Day:", most_likely_states)


change_indices = np.where(np.diff(most_likely_states) != 0)[0] + 1
Ks_init = 300

Ks = []
por = []
for i in range(0, len(change_indices) + 1):
    if i == 0:
        Ks.append(300)
        por.append(Ks_to_por(Ks[i], fc=0.15, wp=0.08))
    elif i == 1:
        Ks.append(250)
        por.append(Ks_to_por(Ks[i], fc=0.15, wp=0.08))
    elif i == 2:
        Ks.append(200)
        por.append(Ks_to_por(Ks[i], fc=0.15, wp=0.08))
    elif i == 3:
        Ks.append(150)
        por.append(Ks_to_por(Ks[i], fc=0.15, wp=0.08))
    else:
        Ks.append(100)
        por.append(Ks_to_por(Ks[i], fc=0.15, wp=0.08))

start_date = datetime(1985, 1, 1)
x = len(most_likely_states)  # Replace with your desired number of days

# Generate a list of dates in the specified format
dates = [(start_date + timedelta(days=i)).strftime('%m/%d/%Y %H:%M') for i in range(x)]

# Print the formatted dates
change_date = []
for date in change_indices:
    change_date.append(dates[date])
    print(change_date)
    

#%%

# Set the working directory
code_directory = r'C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model'
os.chdir(code_directory)

# Function to execute simulation for a given condition state
def run_simulation(model_name, start_date, end_date, porosity, K_sat):
    print(f"Simulating {model_name}")
    print(f"Start Date: {start_date}, End Date: {end_date}")
    
    with open(model_name, 'r') as file:
        lines = file.readlines()

    with open(model_name, 'w') as file:
        in_lid_controls = False  # Flag to track if inside LID_CONTROLS section

        for line in lines:
            # Modify the start and report dates
            if line.startswith('START_DATE'):
                file.write(f"START_DATE            {start_date}\n")
            elif line.startswith("END_DATE"):
                file.write(f"END_DATE             {end_date}\n")
            elif line.startswith("REPORT_START_DATE"):
                file.write(f"REPORT_START_DATE     {start_date}\n")
            elif line.startswith("START_TIME"):
                file.write("START_TIME            00:00:00\n")
            elif line.startswith("REPORT_START_TIME"):
                file.write("REPORT_START_TIME     00:00:00\n")
            
            # Check if inside the LID_CONTROLS section
            elif '[LID_CONTROLS]' in line:
                in_lid_controls = True
                file.write(line)
            
            elif in_lid_controls and line.strip().startswith('[END_LID_CONTROLS]'):
                in_lid_controls = False
                file.write(line)
            
            elif in_lid_controls and 'SOIL' in line:
                # Modify the porosity and K_sat in the SOIL line
                parts = line.split()
                if len(parts) >= 7:  # Ensure the line has enough fields
                    parts[3] = str(porosity)  # Assuming porosity is the 4th value
                    parts[6] = str(K_sat)     # Assuming K_sat is the 7th value
                file.write(' '.join(parts) + '\n')
            
            else:
                # Write the unchanged line
                file.write(line)


#%%
# Loop through the condition states
models = ['basicModel_CS1.inp', 'basicModel_CS2.inp', 'basicModel_CS3.inp', 'basicModel_CS4.inp', 'basicModel_CS5.inp']
for i in range(len(por)):
    print(f"Running condition state: CS {i + 1}")
    
    if i == 0:
        start_date = (datetime.strptime('01/01/1985 00:00', '%m/%d/%Y %H:%M')).strftime('%m/%d/%Y %H:%M')
    else:
        start_date = change_date[i - 1]
    
    if i < len(change_date):
        end_date = (datetime.strptime(change_date[i], '%m/%d/%Y %H:%M') - timedelta(days=1)).strftime('%m/%d/%Y %H:%M')
    else:
        # For the last state, set a fixed end date
        end_date = (datetime.strptime('12/31/2009 23:59', '%m/%d/%Y %H:%M')).strftime('%m/%d/%Y %H:%M')
    
    # Run the simulation
    run_simulation(models[i], start_date, end_date, por[i], Ks[i])
    
#%%
# Define the base name for the SWMM input files
base_name = "basicModel_CS"
report_name = "rpt_basicModel_CS"

# Loop through the files and run simulations
for i in range(1, 6):
    inp_file = f"{base_name}{i}.inp"
    rpt_file = f"{report_name}{i}.rpt"
    #out_file = f"{base_name}{i}.out"  # Default output file

    if os.path.exists(inp_file):
        with Simulation(inputfile=inp_file, reportfile=rpt_file, outputfile=None) as sim:
            print(f"Running simulation for {inp_file}...")
            for step in sim:
                pass
        print(f"Simulation for {inp_file} completed.")
        
        # # Check and remove the .out file if it exists
        # if os.path.exists(out_file):
        #     os.remove(out_file)
        #     print(f"Removed .out file: {out_file}")
    else:
        print(f"File {inp_file} not found.")
        
#%% Remove the out file

for i in range(1, 6):
    out_file = f"{base_name}{i}.out"  # Default output file
    
    if os.path.exists(out_file):
        os.remove(out_file)
        print(f"Removed .out file: {out_file}")
        
#%% Read the LID report files

# Define the file pattern for all five files
file_pattern = "basicModel_CS[1-5].txt"

# List all files matching the pattern
file_list = glob.glob(file_pattern)

# Initialize an empty list to store processed DataFrames
processed_dataframes = []

# Process each file
for file_path in file_list:
    print(f"Processing file: {file_path}")
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start index of the table
    start_index = 0
    for i, line in enumerate(lines):
        if line.startswith('Date        Time'):
            start_index = i
            break

    # Extract the header and data lines
    header = lines[start_index].strip().split()
    data_lines = lines[start_index + 2:]

    # Process the data lines
    data = []
    for line in data_lines:
        if line.strip():
            data.append(line.strip().split())

    # Create a DataFrame
    df = pd.DataFrame(data, columns=header)

    # Convert relevant columns to numeric types
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    # Combine the 'Date' and 'Time' columns into a single datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Drop the original 'Date' and 'Time' columns
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Rename columns
    df.columns = [
        'Elasped_hours', 'Total_inflow_mmh-1', 'Total_evap_mmh-1', 
        'Surf_inf_mmh-1', 'Pav_perc_mmh-1', 'Soil_perc_mmh-1', 
        'Sto_exf_mmh-1', 'Surf_runoff_mmh-1', 'Drain_outflow_mmh-1', 
        'Surf_level_mm', 'Pav_level_mm', 'Soil_moisture_mm', 'Sto_level_mm', 'Datetime'
    ]

    # Set the 'Datetime' column as the index
    df.set_index('Datetime', inplace=True)

    # Fill missing timesteps
    full_datetime_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10T')
    df = df.reindex(full_datetime_index).fillna(0)

    # Add a source column to indicate the file source
    #df['source_file'] = file_path.split('/')[-1]

    # Append the processed DataFrame
    processed_dataframes.append(df)

# Concatenate all processed DataFrames
merged_data = pd.concat(processed_dataframes)

# Save the merged DataFrame to a file
output_path = "merged_completed_dataset.csv"
merged_data.to_csv(output_path)

print("All files processed, missing timesteps filled, and merged dataset saved to:", output_path)


