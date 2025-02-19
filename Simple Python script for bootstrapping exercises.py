import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend for pop-up
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro

# Define parameters
mean = 4.0  # Mean (μ)
std_dev = 1.0  # Standard deviation (σ)
n_samples = 10000  # Number of samples

# Generate samples
samples = np.random.normal(mean, std_dev, n_samples)

# Compute sample mean and standard deviation
computed_mean = np.mean(samples)
computed_std_dev = np.std(samples, ddof=0)

print(f"Ground Truth Mean: {mean}, Computed Mean: {computed_mean}")
print(f"Ground Truth Std Dev: {std_dev}, Computed Std Dev: {computed_std_dev}")

# Compute mean and standard deviation over increasing sample sizes
sample_sizes = np.arange(1, n_samples + 1)
means = np.array([np.mean(samples[:size]) for size in sample_sizes])
std_devs = np.array([np.std(samples[:size], ddof=0) for size in sample_sizes])

# Generate Class A and Class B samples
mean_A, std_dev_A = 3.5, 1.5
mean_B, std_dev_B = 2.1, 2.3
n_class_samples = 100
samples_A = np.random.normal(mean_A, std_dev_A, n_class_samples)
samples_B = np.random.normal(mean_B, std_dev_B, n_class_samples)

# Create subplots to display all plots in one window
fig, axes = plt.subplots(6, 1, figsize=(8, 24))

# Histogram with theoretical PDF
domain = np.linspace(min(samples), max(samples), 100)
pdf_values = stats.norm.pdf(domain, mean, std_dev)
axes[0].hist(samples, bins=30, density=True, alpha=0.7, color='blue', label='Sample Histogram')
axes[0].plot(domain, pdf_values, 'r--', label='Theoretical PDF')
axes[0].set_title('Histogram with Theoretical PDF')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].legend()

# Mean convergence plot
axes[1].plot(sample_sizes, means, label='Computed Mean', color='b')
axes[1].axhline(y=mean, color='r', linestyle='--', label='True Mean')
axes[1].set_xlabel('Number of Samples')
axes[1].set_ylabel('Computed Mean')
axes[1].set_title('Computed Mean vs. Number of Samples')
axes[1].legend()

# Standard deviation convergence plot
axes[2].plot(sample_sizes, std_devs, label='Computed Std Dev', color='g')
axes[2].axhline(y=std_dev, color='r', linestyle='--', label='True Std Dev')
axes[2].set_xlabel('Number of Samples')
axes[2].set_ylabel('Computed Standard Deviation')
axes[2].set_title('Computed Standard Deviation vs. Number of Samples')
axes[2].legend()

# Q-Q Plot
stats.probplot(samples, dist="norm", plot=axes[3])
axes[3].set_title("Q-Q Plot of Samples")

# Histogram of Class A and Class B with PDFs
domain_A = np.linspace(min(samples_A), max(samples_A), 100)
domain_B = np.linspace(min(samples_B), max(samples_B), 100)
pdf_A = stats.norm.pdf(domain_A, mean_A, std_dev_A)
pdf_B = stats.norm.pdf(domain_B, mean_B, std_dev_B)
axes[4].hist(samples_A, bins=15, density=True, alpha=0.6, color='blue', label='Class A')
axes[4].hist(samples_B, bins=15, density=True, alpha=0.6, color='red', label='Class B')
axes[4].plot(domain_A, pdf_A, 'b--', label='PDF A')
axes[4].plot(domain_B, pdf_B, 'r--', label='PDF B')
axes[4].set_title('Histogram of Class A and B with PDFs')
axes[4].set_xlabel('Value')
axes[4].set_ylabel('Density')
axes[4].legend()

# Shapiro-Wilk Test for normality for different sample sizes
sample_sizes_test = [5, 10, 50, 100, 500, 1000, 5000]
p_values = []
for size in sample_sizes_test:
    stat, p_value = shapiro(samples[:size])
    p_values.append(p_value)

# Shapiro-Wilk test p-value plot
axes[5].plot(sample_sizes_test, p_values, marker='o', color='purple', label='p-value')
axes[5].axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
axes[5].set_xlabel('Sample Size')
axes[5].set_ylabel('p-value')
axes[5].set_title('Shapiro-Wilk Test p-value vs. Sample Size')
axes[5].legend()

# Adjust layout and show all plots in one window
plt.tight_layout()
plt.show(block=True)
