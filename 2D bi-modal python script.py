import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Check whether plots are interactive or not ... interactive mode is most useful from the command line
if plt.isinteractive() == True:
    print("Interactive mode is TRUE - exiting.")
    exit()
else:
    print("Interactive mode is FALSE - remember to close graph window.")

# Generate synthetic data for "positive" face (mean closer to one group) and "negative" face (mean closer to another group)
mean_positive = [5.0, 5.0]  # Mean for the "positive" face
cov_positive = [[1, 0], [0, 1]]  # Covariance matrix for positive face

mean_negative = [10.0, 10.0]  # Mean for the "negative" face
cov_negative = [[1, 0], [0, 1]]  # Covariance matrix for negative face

# Generate 500 samples for each face type
samples_positive = np.random.multivariate_normal(mean_positive, cov_positive, 500)
samples_negative = np.random.multivariate_normal(mean_negative, cov_negative, 500)

# Pool all faces into one dataset (combining both positive and negative samples)
all_faces = np.vstack((samples_positive, samples_negative))

# Compute the mean value over all samples in each dimension
actualmean = np.mean(all_faces, axis=0)
print(f"Mean of all faces: {actualmean}")

# Optionally: Calculate Mahalanobis distance between the two means (positive and negative)
cov_matrix = np.cov(all_faces, rowvar=False)  # Covariance matrix of the entire dataset
inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inverse of the covariance matrix

# Mahalanobis distance between the two class means
mahalanobis_distance = distance.mahalanobis(mean_positive, mean_negative, inv_cov_matrix)
print(f"Mahalanobis Distance between Positive and Negative face means: {mahalanobis_distance}")

# Plot scatter plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the positive and negative samples
ax.scatter(samples_positive[:, 0], samples_positive[:, 1], marker=".", color='blue', label='Positive Face')
ax.scatter(samples_negative[:, 0], samples_negative[:, 1], marker=".", color='red', label='Negative Face')

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter plot of Positive and Negative Faces')
ax.set_box_aspect(1.0)
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])
ax.legend()

# Display the plot
plt.show()

exit()
