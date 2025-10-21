import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = r"C:\Users\joshipra\switchdrive\BGI_simulator\04_workspace\SWMM_model"

# Number of folders
n = 5  # Replace with the actual number of folders

# Initialize a list to store data
all_data = []

# Load data from each file
for i in range(1, n + 1):
    folder_name = f"Simulation_{i}"
    file_path = os.path.join(base_dir, folder_name, "cumulative_survivals.csv")
    data = pd.read_csv(file_path, header=None)
    all_data.append(data.values)

# Convert to a 3D NumPy array: shape (n, 4, columns)
all_data = np.array(all_data)

# Calculate mean and standard deviation along axis 0 (across simulations)
mean_values = np.mean(all_data, axis=0)
std_values = np.std(all_data, axis=0)


#%%

ones_row = np.ones((1, mean_values.shape[1]))
zeros_row = np.zeros((1, mean_values.shape[1]))

# Add the row of ones as the fifth row
mean_values_extended = np.vstack((mean_values, ones_row))
std_values_extended = np.vstack((std_values, zeros_row))

#%%
# Plot mean and standard deviation for all rows in one figure
plt.figure(figsize=(10, 6))

# Define colors for the area under the mean
#colors = ['brown', 'green', 'orange', 'red', 'grey']
colors = ['#a6611a', '#dfc27d', '#fdae61', '#80cdc1', '#2c7bb6']

for i in range(5):  # Four rows
    # Plot the area under the mean with different colors
    if i == 0:
        y1 = 0
    else:
        y1 = mean_values_extended[i-1]
    
    plt.fill_between(
        range(mean_values_extended.shape[1]),
        y1,
        mean_values_extended[i],
        alpha=0.15,
        color = colors[i],
        label=f"Row {i+1} Area"        
    )

    # Plot the mean with a solid black line
    plt.plot(mean_values_extended[i], color=colors[i], label=f"Row {i+1} Mean")

    # Plot the standard deviation with black dashed lines
    plt.plot(mean_values_extended[i] - std_values_extended[i], linestyle="--", color=colors[i], label=f"Row {i+1} - Std Dev")
    plt.plot(mean_values_extended[i] + std_values_extended[i], linestyle="--", color=colors[i], label=f"Row {i+1} + Std Dev")



# Customize font only for this plot
# Font settings
font_labels = {'family': 'Calibri', 'size': 20}
# Set y-axis label with custom font
# Set plot title and labels with custom fonts
plt.xlabel("Time [d]", fontdict=font_labels)
plt.ylabel("Survival Probability", fontdict=font_labels)

# Customize the font size of x- and y-tick labels
plt.tick_params(axis='both', which='major', labelsize=16)


# Add grid and adjust layout
plt.grid()
plt.tight_layout()

# Show plot
plt.show()