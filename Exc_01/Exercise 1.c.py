
### Exercise 1.c

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## 1.c.1

# Class A
mean_A = 3.5
std_dev_A = 1.5
n_samples_A = 100

samples_A = np.random.normal(mean_A, std_dev_A, n_samples_A)

## 1.c.2

# Class B
mean_B = 2.1
std_dev_B = 2.3
n_samples_B = 100

samples_B = np.random.normal(mean_B, std_dev_B, n_samples_B)

## 1.c.3

# Define a range that covers both distributions
x = np.linspace(min(min(samples_A), min(samples_B)) - 1, max(max(samples_A), max(samples_B)) + 1, 1000)

# Compute PDFs for both distributions
pdf_A = norm.pdf(x, mean_A, std_dev_A)
pdf_B = norm.pdf(x, mean_B, std_dev_B)

# Plot histograms
plt.hist(samples_A, bins=30, density=True, alpha=0.6, color='blue', label='Class A Histogram')
plt.hist(samples_B, bins=30, density=True, alpha=0.6, color='green', label='Class B Histogram')

# Plot PDFs
plt.plot(x, pdf_A, 'b-', linewidth=2, label='Class A PDF')
plt.plot(x, pdf_B, 'g-', linewidth=2, label='Class B PDF')

# Labels and legend
plt.title('Histograms and PDFs of Class A and Class B')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

## 1.c.4, 1.c.5

value = 2.8

# Calculate PDF values at 2.8 for both classes
pdf_A = norm.pdf(value, mean_A, std_dev_A)
pdf_B = norm.pdf(value, mean_B, std_dev_B)

print(f"PDF at {value} for Class A: {pdf_A:.4f}")
print(f"PDF at {value} for Class B: {pdf_B:.4f}")

# Determine more likely class
if pdf_A > pdf_B:
    print(f"A score of {value} is more likely to belong to Class A.")
else:
    print(f"A score of {value} is more likely to belong to Class B.")

## 1.c.7

# Sample values to classify
test_values = [2.8, 3.0, 1.8]

for x in test_values:
    z_A = (x - mean_A) / std_dev_A
    z_B = (x - mean_B) / std_dev_B

    print(f"Score: {x}")
    print(f"Z-score for Class A: {z_A:.2f}")
    print(f"Z-score for Class B: {z_B:.2f}")

    if abs(z_A) < abs(z_B):
        print(f"→ Score {x} is closer to Class A.\n")
    else:
        print(f"→ Score {x} is closer to Class B.\n")


