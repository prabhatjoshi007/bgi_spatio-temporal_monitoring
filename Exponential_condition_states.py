# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:41:48 2024

@author: joshipra
"""

import numpy as np
import matplotlib.pyplot as plt

def exponential_survival(theta, lambda_j):
    """
    Exponential survival function: S_j(theta) = exp(-theta/lambda_j)
    """
    return np.exp(-theta / lambda_j)

def exponential_pdf(theta, lambda_j):
    """
    Exponential PDF derived from the survival function.
    f(theta) = -dS(theta)/dtheta
    """
    return (1 / lambda_j) * np.exp(-theta / lambda_j)

# Parameters for the four condition states
lambdas = [5, 10, 15, 20]  # Ensure increasing scale parameters
theta_values = np.linspace(0, 25,  25*365)  # Values for theta over 25 years

# Exclude theta = 0 for validation
theta_positive = theta_values[theta_values > 0]

# Calculate survival functions and PDFs
survival_functions = []
pdfs = []
for j, lambda_j in enumerate(lambdas):
    S_j = exponential_survival(theta_values, lambda_j)
    survival_functions.append(S_j)
    pdf_j = exponential_pdf(theta_values, lambda_j)
    pdfs.append(pdf_j)

# Validate condition S_j(theta) > S_(j-1)(theta) for theta > 0 and j = 2, 3, 4
for j in range(1, len(survival_functions)):
    assert np.all(survival_functions[j][theta_values > 0] > survival_functions[j - 1][theta_values > 0]), \
        f"Condition violated: S_{j+1}(theta) <= S_{j}(theta) for theta > 0"

# Plot survival functions
plt.figure(figsize=(10, 6))
for j, S_j in enumerate(survival_functions, start=1):
    plt.plot(theta_values, S_j, label=f"Survival Function S_{j}(θ)")

plt.xlabel("Age (θ) (years)")
plt.ylabel("Survival Probability")
plt.title("Exponential Survival Functions for Condition States")
plt.legend()
plt.grid(True)
plt.show()

# Plot probability density functions
plt.figure(figsize=(10, 6))
for j, pdf_j in enumerate(pdfs, start=1):
    plt.plot(theta_values, pdf_j, label=f"PDF of S_{j}(θ)")

plt.xlabel("Age (θ) (years)")
plt.ylabel("Probability Density")
plt.title("Probability Density Functions of Exponential Survival Functions")
plt.legend()
plt.grid(True)
plt.show()

print("All conditions S_j(theta) > S_(j-1)(theta) are satisfied for θ > 0.")
