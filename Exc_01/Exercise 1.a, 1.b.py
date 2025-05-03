
### Exercise 1.a

import numpy as np
import matplotlib.pyplot as plt

## 1.a.1

# Generate 1000 samples from a normal distribution
mean = 4.0
std_dev = 1.0
n_samples = 10
samples = np.random.normal(mean, std_dev, n_samples)

## 1.a.2

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Histogram of 1000 Samples from Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')

#plt.show will halt thread - program cannot proceed until open windows have been closed
plt.show()

## 1.a.3

#compute the mean of the samples in over the first 10, the first 20, the first 30,... samples
step_count = 0
step_size = 50
mean_values = np.zeros(np.uint(n_samples/step_size))
for end_index in range(step_size, n_samples + 1, step_size):
    mean_values[step_count] = np.mean(samples[0:end_index])
    step_count = step_count + 1

plt.plot(mean_values)
plt.xlabel("Number of samples included in mean, times {}".format(step_size))
plt.ylabel('Mean over samples')
plt.show()

# Compute sample mean and standard deviation, compare with ground truth
computed_mean = np.mean(samples)
computed_std_dev = np.std(samples)

mean_error_percent = abs(mean - computed_mean) / mean * 100
std_dev_error_percent = abs(std_dev - computed_std_dev) / std_dev * 100

print(f"Mean error (%): {mean_error_percent:.2f}%")
print(f"Standard deviation error (%): {std_dev_error_percent:.2f}%")

## 1.a.4

# Desired error percentage
desired_error_percent = 0.1

required_sem = desired_error_percent * mean / 100
required_n = (std_dev / required_sem) ** 2

print(f"Required sample size for {desired_error_percent}% error: {int(required_n)}")

## 1.a.5

# Compute the mean as a function of the number of samples
mean_values = np.zeros(n_samples)
for i in range(1, n_samples + 1):
    mean_values[i - 1] = np.mean(samples[:i])

# Plot the computed mean
plt.plot(range(1, n_samples + 1), mean_values)
plt.xlabel('Number of samples used to compute mean')
plt.ylabel('Computed mean')
plt.title('Computed Mean as a Function of Number of Samples')
plt.show()

## 1.a.6

# Compute the standard deviation as a function of the number of samples
std_dev_values = np.zeros(n_samples)
for i in range(1, n_samples + 1):
    std_dev_values[i - 1] = np.std(samples[:i])

# Plot the computed standard deviation
plt.plot(range(1, n_samples + 1), std_dev_values)
plt.xlabel('Number of samples used to compute standard deviation')
plt.ylabel('Computed standard deviation')
plt.title('Computed Standard Deviation as a Function of Number of Samples')
plt.show()

### Exercise 1.b

from scipy.stats import norm
import scipy.stats as stats
from scipy.stats import shapiro

### 1.b.1

# Generate x values across the sample range
x = np.linspace(min(samples), max(samples), 1000)

# Compute the theoretical PDF using the known mean and std_dev
pdf = norm.pdf(x, mean, std_dev)

# Plot the histogram again
plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue', label='Sample Histogram')

# Overlay the theoretical PDF
plt.plot(x, pdf, 'r-', linewidth=2, label='Theoretical PDF')

plt.title('Histogram and Theoretical PDF')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show()

### 1.b.2

# Create a Q-Q plot to compare the sample data with a normal distribution
stats.probplot(samples, dist="norm", sparams=(mean, std_dev), plot=plt)

# Add title and labels
plt.title('Q-Q Plot for Samples')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

# Show the Q-Q plot
plt.show()

### 1.b.4

# Perform Shapiro-Wilk test
w_stat, p_value = shapiro(samples)

print(f"Shapiro-Wilk Test Statistic: {w_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
if p_value > 0.05:
    print("The data likely follows a normal distribution \n(fail to reject H₀).")
else:
    print("The data does not follow a normal distribution \n(reject H₀).")


