### Exercise 2.a

import numpy as np
import matplotlib.pyplot as plt

## 2.a.1

# Generate 1000 samples from a 2D normal distribution
mean = [5.0, 8.0]
covariance_matrix = [[1, 0],
                     [0, 1]]

# 1000 rows, 2 columns - each row a sample, each column a dimension
samples = np.random.multivariate_normal(mean, covariance_matrix, 1000)

## 2.a.2, 2.a.3

# Plot scatter plot
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(samples[:,0], samples[:,1], marker=".", color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_box_aspect(1.0)
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])


# plt.show will halt thread - program cannot terminate until open windows have been closed
# plt.figure(figsize=(10,10))
plt.show()

## 2.a.4

# Compute actual mean and covariance from the generated samples
# Option 0 signifies that we want mean computed along first dimension (rows, i.e., samples)
actual_mean = np.mean(samples,0)
cov = np.cov(samples, rowvar=False)

# Inverse covariance matrix
inv_cov = np.linalg.inv(cov)

# Mahalanobis distance
def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))

# Assign color based on Mahalanobis distance for each sample
colors = []
threshold = 1.96  # 95% confidence boundary in 2D

for sample in samples:
    dist = mahalanobis_distance(sample, actual_mean, inv_cov)
    if dist < threshold:
        colors.append('green')
    else:
        colors.append('red')

# Plot scatter plot with colors based on Mahalanobis distance
fig2, ax2 = plt.subplots(figsize=(6, 6))

ax2.scatter(samples[:, 0], samples[:, 1], c=colors, marker=".")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_box_aspect(1.0)
ax2.set_xlim([0, 20])
ax2.set_ylim([0, 20])
plt.title("Mahalanobis Distance Color Coding (<1.96 = Green)")

plt.show()

## 2.a.5

# Count how many samples are green (i.e., within the Mahalanobis distance threshold)
num_within = colors.count('green')
num_total = len(colors)
proportion_within = num_within / num_total

print(f"Proportion of samples within Mahalanobis distance threshold: {proportion_within:.4f}")



