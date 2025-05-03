### Exercise 2.b

import numpy as np
import matplotlib.pyplot as plt

## 2.b.1, 2.b.2, 2.b.3

# Generate 1000 samples from a 2D normal distribution for Class A: centered lower left, circular shape
mean_A = [4.0, 6.0]
covariance_matrix_A = [[1.0, 0.0],
                       [0.0, 1.0]]

# 1000 rows, 2 columns - each row a sample, each column a dimension for Class A
samples_A = np.random.multivariate_normal(mean_A, covariance_matrix_A, 1000)

# Generate 1000 samples from a 2D normal distribution for Class B: centered upper right, elongated and rotated
mean_B = [10.0, 12.0]
covariance_matrix_B = [[3.0, 1.5],
                       [1.5, 3.0]]

# 1000 rows, 2 columns - each row a sample, each column a dimension for Class B
samples_B = np.random.multivariate_normal(mean_B, covariance_matrix_B, 1000)

## 2.b.4

# Plot samples from both classes
fig1, ax1 = plt.subplots(figsize=(6, 6))

# Plot Class A samples in black
ax1.scatter(samples_A[:, 0], samples_A[:, 1], color='black', marker='.', label='Class A', alpha=0.5)

# Plot Class B samples in black
ax1.scatter(samples_B[:, 0], samples_B[:, 1], color='black', marker='x', label='Class B', alpha=0.5)

# Axis labels and aspect ratio
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Class A and B Samples')
ax1.set_box_aspect(1.0)
ax1.set_xlim([0, 20])
ax1.set_ylim([0, 20])
ax1.legend()

plt.show()

## 2.b.5

# Generate 1000 samples from a 2D uniform distribution
num_uniform_samples = 1000
uniform_samples = np.random.uniform(low=[0, 0], high=[20, 20], size=(num_uniform_samples, 2))

## 2.b.6

# Compute means and covariances from the samples
mean_A_actual = np.mean(samples_A, axis=0)
cov_A_actual = np.cov(samples_A, rowvar=False)
inv_cov_A = np.linalg.inv(cov_A_actual)

mean_B_actual = np.mean(samples_B, axis=0)
cov_B_actual = np.cov(samples_B, rowvar=False)
inv_cov_B = np.linalg.inv(cov_B_actual)

# Mahalanobis function
def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))

# Classify uniform samples
classified_colors = []

for point in uniform_samples:
    dist_to_A = mahalanobis_distance(point, mean_A_actual, inv_cov_A)
    dist_to_B = mahalanobis_distance(point, mean_B_actual, inv_cov_B)

    if dist_to_A < dist_to_B:
        classified_colors.append('green')  # Closer to Class A
    else:
        classified_colors.append('red')  # Closer to Class B

# Plot classified uniform samples
fig2, ax2 = plt.subplots(figsize=(6, 6))

# Plot the uniform samples (classified)
ax2.scatter(uniform_samples[:, 0], uniform_samples[:, 1], c=classified_colors, marker='.', alpha=0.4, label='Uniform samples')

# Overlay Class A and B samples in black
ax2.scatter(samples_A[:, 0], samples_A[:, 1], color='black', marker='.', alpha=0.6, label='Class A')
ax2.scatter(samples_B[:, 0], samples_B[:, 1], color='black', marker='x', alpha=0.6, label='Class B')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Mahalanobis Classification of Uniform Samples')
ax2.set_box_aspect(1.0)
ax2.set_xlim([0, 20])
ax2.set_ylim([0, 20])
ax2.legend()

plt.show()


