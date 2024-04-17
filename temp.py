import numpy as np
import matplotlib.pyplot as plt

def sector_sampling(x_center, y_center, direction_vector, min_radius, max_radius, num_arcs, num_points_per_arc):
    all_xs, all_ys = [], []
    
    # Calculate the angles for the sector
    angle_center = np.arctan2(direction_vector[1], direction_vector[0])
    min_angle = angle_center - np.pi / 4
    max_angle = angle_center + np.pi / 4
    
    # Generate angles for each arc
    all_xs, all_ys = [], []
    radii = np.linspace(min_radius, max_radius, num_arcs)
    for radius in radii:
        angles = np.linspace(min_angle, max_angle, num_points_per_arc, endpoint=False)
        xs = np.round(x_center + radius * np.cos(angles)).astype(int)
        ys = np.round(y_center + radius * np.sin(angles)).astype(int)
        all_xs.extend(xs)
        all_ys.extend(ys)
    return all_xs, all_ys


def plot_sector_with_samples(x_center, y_center, direction_vector, min_radius, max_radius, num_arcs, num_points_per_arc):
    xs, ys = sector_sampling(x_center, y_center, direction_vector, min_radius, max_radius, num_arcs, num_points_per_arc)
    plt.scatter(xs, ys, color='r')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sector with Polar Sampling')
    plt.show()

# Example usage
x_center = 0
y_center = 0
direction_vector = [1, 1]  # Example direction vector
min_radius = 10
max_radius = 50
num_arcs = 9
num_points_per_arc = 20
plot_sector_with_samples(x_center, y_center, direction_vector, min_radius, max_radius, num_arcs, num_points_per_arc)
