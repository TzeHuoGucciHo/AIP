
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 samples from a normal distribution
mean = 4.0
std_dev = 1.0
n_samples = 1000
samples = np.random.normal(mean, std_dev, n_samples)

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Histogram of 1000 Samples from Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')

#plt.show will halt thread - program cannot proceed until open windows have been closed
plt.show()

#compute the mean of the samples in over the first 10, the first 20, the first 30,... samples
step_count = 0
step_size = 10
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

# Estimate samples needed to achieve a certain error

# Desired error percentage
desired_error_percent = 0.1

required_sem = desired_error_percent * mean / 100
required_n = (std_dev / required_sem) ** 2

print(f"Required sample size for {desired_error_percent}% error: {int(required_n)}")


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

exit()