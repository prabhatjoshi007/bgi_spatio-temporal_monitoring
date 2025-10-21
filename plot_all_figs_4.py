# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:09:21 2025

@author: joshipra
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Set the base directory and number of simulations
base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"

static_folder = "Simulation_0"
static_file_path = os.path.join(base_dir, static_folder, "catchment_outfall_static.csv")
static_outfall = pd.read_csv(static_file_path)
static_outfall['Time'] = pd.to_datetime(static_outfall['Time'])

n = 5  # Number of simulations

# Initialize lists to store data
outfall_data = []

start_time = "01/01/1985 00:00"
end_time = "12/31/2009 23:59"
full_datetime_index_full = pd.date_range(start=start_time, end=end_time, freq="10min")


# Step 2: Process each dataset and pad arrays
for i in range(1, n + 1):
    folder_name = f"Simulation_{i}"
    file_path = os.path.join(base_dir, folder_name, f"merged_outfall_data_{i}.csv")
    outfall = pd.read_csv(file_path)
    outfall['Time'] = pd.to_datetime(outfall['Time'])
    
    outfall_volume = outfall["Outfall_Volume"].values

    # Append padded arrays
    plt.plot(outfall_volume)
    outfall_data.append(outfall_volume)

# Convert lists to 2D NumPy arrays

outfall["Time"] = pd.to_datetime(outfall["Time"])

# Calculate mean and standard deviation (ignore NaN values)
to_m3 = 10*60/1000
static_outfall2 = static_outfall["Outfall_Volume"] * to_m3
outfall_mean = np.nanmean(outfall_data, axis=0) * to_m3
outfall_std = np.nanstd(outfall_data, axis=0) * to_m3

cumulative_variance = np.cumsum(outfall_std**2)  # Variance is SD^2
cumulative_sd = np.sqrt(cumulative_variance)  # Cumulative SD is sqrt of cumulative variance
cumulative_sd_scale = cumulative_sd/1e5

# Plot results
fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

# Surf Runoff Plot
axes.plot(outfall["Time"], (outfall_mean/1e5).cumsum(), label = 'Mean', linewidth = 2, color = 'g') 
axes.fill_between(
    outfall["Time"],
    (outfall_mean/1e5).cumsum() - cumulative_sd_scale,
    (outfall_mean/1e5).cumsum() + cumulative_sd_scale,
    alpha=0.2, label="± 1 SD", color = 'g'
)
axes.plot(static_outfall["Time"], (static_outfall2/1e5).cumsum(), 'k--', linewidth = 2, label = 'Stationary/CS1')

# Customize font only for this plot
font_props = {'family': 'Calibri', 'size': 25}

# Set y-axis label with custom font
axes.set_ylabel("Cumulative CSO volume [10^5 m³]", fontdict=font_props)
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


