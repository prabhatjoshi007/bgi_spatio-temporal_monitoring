# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:49:48 2024

@author: joshipra
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_states = 5         # Number of condition states (CS1 to CS5)
planning_horizon = 25  # Planning horizon of 50 years
initial_state = 0      # Start in the "Excellent" state (CS1)

# Log-normal parameters for each transition (mean and standard deviation)
# These parameters represent the log-normal distribution for the sojourn time in each state
lognormal_params = [
    (1.5, 0.5),  # CS1: "Excellent" to CS2
    (1.2, 0.4),  # CS2: "Good" to CS3
    (1.0, 0.3),  # CS3: "Moderate" to CS4
    (0.8, 0.2),  # CS4: "Poor" to CS5
]

# Transition matrix for five states (transitions to worse condition or staying in the current state)
# P[i][j] represents the probability of transitioning from state i to state j
transition_matrix = np.array([
    [0.7, 0.3, 0.0, 0.0, 0.0],  # CS1
    [0.0, 0.6, 0.4, 0.0, 0.0],  # CS2
    [0.0, 0.0, 0.5, 0.5, 0.0],  # CS3
    [0.0, 0.0, 0.0, 0.4, 0.6],  # CS4
    [0.0, 0.0, 0.0, 0.0, 1.0],  # CS5 (absorbing state)
])

# Initialize variables
years = np.arange(planning_horizon + 1)
states_over_time = [initial_state]
current_state = initial_state
current_year = 0

# Function to sample sojourn time from a log-normal distribution
def sample_sojourn_time(mean, sigma):
    return int(np.random.lognormal(mean, sigma))

# Simulate over the planning horizon
while current_year < planning_horizon and current_state < num_states - 1:
    # Get log-normal parameters for the current state
    if current_state < len(lognormal_params):
        mean, sigma = lognormal_params[current_state]
        sojourn_time = sample_sojourn_time(mean, sigma)
    else:
        sojourn_time = 1  # Default to 1 year if we're in the last state (absorbing)

    # Advance the simulation based on sojourn time and update state
    for _ in range(sojourn_time):
        if current_year >= planning_horizon:
            break
        states_over_time.append(current_state)
        current_year += 1

    # Transition to the next state based on the transition matrix
    if current_state < num_states - 1:
        current_state = np.random.choice(range(num_states), p=transition_matrix[current_state])

# Extend the final state to the end of the planning horizon if the simulation ends early
if current_year < planning_horizon:
    states_over_time.extend([current_state] * (planning_horizon - current_year))

# Plot the results
plt.figure(figsize=(10, 6))
plt.step(years, states_over_time, where='post')
plt.xlabel('Year')
plt.ylabel('Condition State')
plt.title('Condition State Over 50-Year Planning Horizon')
plt.yticks(range(num_states), ["CS1 (Excellent)", "CS2 (Good)", "CS3 (Moderate)", "CS4 (Poor)", "CS5 (Failed)"])
plt.grid()
plt.show()
