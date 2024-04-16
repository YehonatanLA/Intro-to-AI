import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Define the points and their classifications
points = np.array([
    [1, 5],
    [6, 2],
    [7, 3],
    [8, 2],
    [9, 1],
    [2, 7],
    [3, 8],
    [5, 1],
    [6, 2],
    [7, 3],
    [8, 4],
    [9, 5],
    [7, 2],
    [8, 3]
])

classifications = [
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1,
    0, 0
]

# Create the Voronoi diagram
vor = Voronoi(points)

# Classify each point in the Voronoi diagram
classified_points = []
for point in vor.vertices:
    nearest_distance = float('inf')
    nearest_classification = None
    for p, classification in zip(points, classifications):
        distance = np.linalg.norm(point - p)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_classification = classification
    classified_points.append(nearest_classification)

# Plot the Voronoi diagram
fig, ax = plt.subplots(figsize=(8, 6))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black')

# Color the regions based on the classifications
for region in vor.regions:
    if -1 not in region and region:
        polygon = [vor.vertices[i] for i in region]
        classification = classified_points[region[0]]
        if classification == 1:
            plt.fill(*zip(*polygon), color='red', alpha=0.5)
        elif classification == 0:
            plt.fill(*zip(*polygon), color='blue', alpha=0.5)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('1-NN Classifier (Voronoi Diagram)')

# Show the plot
plt.show()
