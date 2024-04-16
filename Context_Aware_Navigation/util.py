import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_nodes_coord(nodes, suffix=""):
    plt.cla()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.scatter(nodes[:, 0], 480 - nodes[:, 1], s=1)
    plt.savefig(f"tmp/node_coords_{suffix}.png")


def plot_frontiers(front, suffix=""):
    plt.cla()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ar = np.array(front)
    ax.scatter(ar[:, 0], 480 - ar[:, 1], s=1)
    plt.savefig(f"tmp/frontier_{suffix}.png")


def plot_frontiers_new_obs(observed_frontiers, new_frontiers):
    plt.cla()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.scatter(observed_frontiers[:, 0], 480 - observed_frontiers[:, 1], s=1, c="r", label="observed")
    ax.scatter(new_frontiers[:, 0], 480 - new_frontiers[:, 1], s=1, c="b", label="new")
    plt.legend()
    plt.savefig("tmp/frontier_new_obs.png")



def generate_cost_map(map_data: cv2.Mat, obstacle_cost=100, inflation_radius=3, inflation_scale=1.0):
    """
    map_data: RGB image
    """
    mask = np.all(map_data == (127, 127, 127), axis=-1)
    map_data[mask] = (0, 0, 0)
    mask = ~mask
    map_data[mask] = (255, 255, 255)
    map_bin = cv2.cvtColor(map_data, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("tmp/1_bin.png", map_bin)
    distance_map = cv2.distanceTransform(map_bin, cv2.DIST_L2, 5)
    norm_distance_map = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("tmp/1_dist.png", norm_distance_map)

    # Create a cost map based on the distance transform
    cost_map = obstacle_cost * np.exp(-inflation_scale * distance_map / inflation_radius)
    normalized_cost_map = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_cost_map
