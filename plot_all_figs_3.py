# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:09:21 2025

@author: joshipra
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# Set the base directory and number of simulations
base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"

static_folder = "Simulation_0"
static_file_path = os.path.join(base_dir, static_folder, "catchment_flood_static.csv")
static_flood = pd.read_csv(static_file_path)
static_flood['Time'] = pd.to_datetime(static_flood['Time'])

n = 5  # Number of simulations

# Initialize lists to store data
flood_data = []

start_time = "01/01/1985 00:00"
end_time = "12/31/2009 23:59"
full_datetime_index_full = pd.date_range(start=start_time, end=end_time, freq="10min")

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
    file_path = os.path.join(base_dir, folder_name, f"merged_flood_data_{i}.csv")
    flood = pd.read_csv(file_path)
    flood['Time'] = pd.to_datetime(flood['Time'])
    
    #flood['Year'] = flood['Time'].dt.year
    #flood['Flood_Volume'] = flood['Flood_Volume']
    
    #annual_flood_volume = flood.groupby('Year')['Flood_Volume'].sum()
    #annual_flood_volume_df = annual_flood_volume.reset_index()

    # Extract columns
    
    flood_volume = flood["Flood_Volume"].values

    # # Pad arrays only if necessary
    # surf_runoff_padded = np.pad(surf_runoff, (0, max(0, max_length - len(surf_runoff))), constant_values=0)
    # drain_outflow_padded = np.pad(drain_outflow, (0, max(0, max_length - len(drain_outflow))), constant_values=0)

    # Append padded arrays
    flood_data.append(flood_volume)

# Convert lists to 2D NumPy arrays

flood["Time"] = pd.to_datetime(flood["Time"])

# Calculate mean and standard deviation (ignore NaN values)
to_m3 = 10*60/1000
static_flood2 = static_flood["Flood_Volume"] * to_m3
flood_mean = np.nanmean(flood_data, axis=0) * to_m3
flood_std = np.nanstd(flood_data, axis=0) * to_m3

cumulative_variance = np.cumsum(flood_std**2)  # Variance is SD^2
cumulative_sd = np.sqrt(cumulative_variance)  # Cumulative SD is sqrt of cumulative variance
cumulative_sd_scale = cumulative_sd/1000

# Plot results
fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

# Surf Runoff Plot
axes.plot(flood["Time"], (flood_mean/1000).cumsum(), label = 'Mean', linewidth = 2) 
axes.fill_between(
    flood["Time"],
    (flood_mean/1000).cumsum() - cumulative_sd_scale,
    (flood_mean/1000).cumsum() + cumulative_sd_scale,
    alpha=0.2, label="± 1 SD"
)
axes.plot(static_flood["Time"], (static_flood2/1000).cumsum(), 'k--', linewidth = 2, label = 'Stationary/CS1')

# Customize font only for this plot
font_props = {'family': 'Calibri', 'size': 25}

# Set y-axis label with custom font
axes.set_ylabel("Cumulative flood volume [10^3 m³]", fontdict=font_props)
axes.set_xlabel("Time [year]", fontdict=font_props)


# Increase font size for tick labels
axes.tick_params(axis='x', labelsize=20)  # X-axis tick label font size
axes.tick_params(axis='y', labelsize=20)  # Y-axis tick label font size

# Customize legend font
legend = axes.legend()
for text in legend.get_texts():
    text.set_fontfamily('Calibri')
    text.set_fontsize(25)

plt.grid()
plt.tight_layout()


plt.show()


