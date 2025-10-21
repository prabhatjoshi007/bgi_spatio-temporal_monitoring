import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Set the base directory and number of simulations
base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"
n = 5  # Number of simulations

# Initialize lists to store data
surf_runoff_data = []
drain_outflow_data = []

# Step 1: Determine the maximum length
# max_length = 0
# for i in range(1, n + 1):
#     folder_name = f"Simulation_{i}"
#     file_path = os.path.join(base_dir, folder_name, f"merged_completed_dataset_{i}.csv")
#     data = pd.read_csv(file_path)

#     # Update max_length
#     max_length = max(max_length, len(data))

# Step 2: Process each dataset and pad arrays
for i in range(1, n + 1):
    folder_name = f"Simulation_{i}"
    file_path = os.path.join(base_dir, folder_name, f"merged_completed_dataset_{i}.csv")
    BGI_data = pd.read_csv(file_path)

    # Extract columns
    
    surf_runoff = BGI_data["Surf_runoff_mmh-1"].values
    drain_outflow = BGI_data["Drain_outflow_mmh-1"].values

    # # Pad arrays only if necessary
    # surf_runoff_padded = np.pad(surf_runoff, (0, max(0, max_length - len(surf_runoff))), constant_values=0)
    # drain_outflow_padded = np.pad(drain_outflow, (0, max(0, max_length - len(drain_outflow))), constant_values=0)

    # Append padded arrays
    surf_runoff_data.append(surf_runoff)
    drain_outflow_data.append(drain_outflow)


# Convert lists to 2D NumPy arrays
surf_runoff_data = np.array(surf_runoff_data)
drain_outflow_data = np.array(drain_outflow_data)
time = BGI_data["Datetime"]

# Calculate mean and standard deviation (ignore NaN values)
BGI_area = 5000 #m2
to_m3 = 1/6/1000

surf_runoff_mean = np.nanmean(surf_runoff_data, axis=0) * BGI_area * to_m3
surf_runoff_std = np.nanstd(surf_runoff_data, axis=0) * BGI_area * to_m3
drain_outflow_mean = np.nanmean(drain_outflow_data, axis=0) * BGI_area * to_m3
drain_outflow_std = np.nanstd(drain_outflow_data, axis=0) * BGI_area * to_m3

cumulative_variance_surf_runoff = np.cumsum(surf_runoff_std**2)  # Variance is SD^2
cumulative_sd_surf_runoff = np.sqrt(cumulative_variance_surf_runoff)  # Cumulative SD is sqrt of cumulative variance

cumulative_variance_drain_outflow = np.cumsum(drain_outflow_std**2)  # Variance is SD^2
cumulative_sd_drain_outflow = np.sqrt(cumulative_variance_drain_outflow)  # Cumulative SD is sqrt of cumulative variance

#%%
# Plot results
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Surf Runoff Plot
axes[0].plot(time, surf_runoff_mean.cumsum()/1e5, label="Mean", color = 'g')
axes[0].fill_between(
    range(len(surf_runoff_mean)),
    #time,
    surf_runoff_mean.cumsum()/1e5 - cumulative_sd_surf_runoff/1e5,
    surf_runoff_mean.cumsum()/1e5 + cumulative_sd_surf_runoff/1e5,
    alpha=0.2, label="SD",
    color = 'y'
)
axes[0].set_ylabel("Surface Runoff [1e5 m3]")
axes[0].legend()

# Drain Outflow Plot
axes[1].plot(drain_outflow_mean.cumsum()/1e5, label="Mean", color = 'b')
axes[1].fill_between(
    range(len(drain_outflow_mean)),
    #time,
    drain_outflow_mean.cumsum()/1e5 - cumulative_sd_drain_outflow/1e5,
    drain_outflow_mean.cumsum()/1e5 + cumulative_sd_drain_outflow/1e5,
    alpha=0.2, label="SD",
    color = 'r'
)
axes[1].set_ylabel("Drain Outflow [1e5 m3]")
axes[1].set_xlabel("Time [10 min]")
axes[1].legend()

plt.tight_layout()
plt.show()

#%%
high_vol_start = '1987-09-25 10:20:00'
high_vol_end = '1987-09-27 11:30:00'

high_dur_start = '2001-01-05 04:40:00,2001-01-06 23:40:00'
high_dur_end = '2001-01-05 04:40:00,2001-01-06 23:40:00'

high_int_start = '1988-05-10 13:10:00'
high_int_end = '1988-05-11 08:00:00'