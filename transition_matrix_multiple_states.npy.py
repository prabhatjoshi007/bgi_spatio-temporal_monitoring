# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 23:57:08 2025

@author: Prabhat Joshi
"""
import numpy as np
from scipy.signal import fftconvolve
from compute_k_lamda import compute_k_lamda
import matplotlib.pyplot as plt


def weibull_pdf(q, lambda_j, k_j):
    """Weibull probability density function."""
    return (k_j / lambda_j) * (q / lambda_j) ** (k_j - 1) * np.exp(- (q / lambda_j) ** k_j)

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
    Compute transition probability matrices for each time step, allowing jumps beyond adjacent states.
    """
    # Time vector
    q = np.linspace(0, num_years, num_years * 365)  # Daily timesteps
    delta = num_years / (num_years * 365)  # Time step size

    # Compute PDFs and survival functions for each state
    pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
    cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
    cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)

    # Initialize list to hold transition matrices
    transition_matrices = []

    # Compute transition matrices for each time step
    for x in range(num_years * 365):
        P = np.zeros((num_states + 1, num_states + 1))  # Transition matrix

        for j in range(num_states):
            total_prob = 0
            remaining_survival = cumulative_survivals[j][x] if j == 0 else cumulative_survivals[j][x] - cumulative_survivals[j - 1][x]
            
            for next_state in range(j + 1, num_states + 1):
                num = cumulative_pdfs[next_state - 1][x]
                den = remaining_survival
                if den > 0 and num > 0:
                    P[j, next_state] = min(1 - total_prob, num * delta / den)
                    total_prob += P[j, next_state]
            
            # Probability of remaining in state j
            P[j, j] = max(0, 1 - total_prob)

        # Final absorbing state remains unchanged
        P[-1, -1] = 1

        transition_matrices.append(P)

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

k = [2.042, 2.042, 2.042, 2.042]
lambdas = [2.393, 3.590, 4.188, 3.590]
delta = 1/365              
num_years = 25
num_days = 365 * num_years        
num_states = 4
q =  np.linspace(0, num_years, num_days)


pdfs = [weibull_pdf(q, lambdas[j], k[j]) for j in range(num_states)]
cumulative_pdfs = compute_cumulative_pdfs(pdfs, q, delta)
cumulative_survivals = compute_cumulative_survivals(cumulative_pdfs, q, delta)
transition_matrices = compute_transition_probability_matrices(num_states, num_years, lambdas, k)

plot_cumulative_survivals(cumulative_survivals, q)

plot_cumulative_pdfs(cumulative_pdfs, q)


# Storage for state vectors over time
state_vector = np.array([1, 0, 0, 0, 0])  # Initial state
state_vectors = np.zeros((num_days, len(state_vector)))
state_vectors[0] = state_vector
state_progression = np.zeros(num_days, dtype=int)  # Tracks the actual state over time


# Iterate through each timestep
for t in range(1, num_days):
    #print("Old state vector:", state_vector)
    state_vector = np.dot(state_vector, transition_matrices[t])
    #print("New state vector:", state_vector)
    state_vectors[t] = state_vector
    
    

for tt in range(1, num_days):  # Use weeks instead of days if updated
    state_progression[0] = 0  # Initial state is always 0

    current_state = state_progression[tt - 1]  # Get previous state

# Zero out probabilities of lower states
    allowed_probabilities = state_vectors[tt].copy()
    allowed_probabilities[:current_state] = 0  # Disallow lower states

# Normalize the remaining probabilities to sum to 1
    if allowed_probabilities.sum() > 0:
        allowed_probabilities /= allowed_probabilities.sum()
    else:
        allowed_probabilities[current_state] = 1  # Stay in the current state if no valid transition

    # Sample next state from the adjusted probability distribution
    state_progression[tt] = np.random.choice(len(state_vector), p=allowed_probabilities)

print("Sum of states: ", sum(state_progression))