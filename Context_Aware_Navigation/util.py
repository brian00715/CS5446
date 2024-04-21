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
    # cv2.imwrite("tmp/1_bin.png", map_bin)
    distance_map = cv2.distanceTransform(map_bin, cv2.DIST_L2, 5)
    norm_distance_map = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imwrite("tmp/1_dist.png", norm_distance_map)

    # Create a cost map based on the distance transform
    cost_map = obstacle_cost * np.exp(-inflation_scale * distance_map / inflation_radius)
    normalized_cost_map = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_cost_map


def calcu_ave_curvature(trajectory):
    trajectory = np.array(trajectory)

    # 计算轨迹点的数量
    n_points = len(trajectory)

    # 初始化曲率数组
    curvatures = []

    # 遍历轨迹点,计算每个点的曲率
    for i in range(1, n_points - 1):
        # 获取当前点及其前后相邻点的坐标
        x0, y0 = trajectory[i - 1]
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]

        # 计算一阶差分
        dx1 = x1 - x0
        dy1 = y1 - y0
        dx2 = x2 - x1
        dy2 = y2 - y1

        # 计算二阶差分
        d2x = x2 - 2 * x1 + x0
        d2y = y2 - 2 * y1 + y0

        # 计算曲率
        numerator = abs(dx1 * d2y - d2x * dy1)
        denominator = (dx1**2 + dy1**2) ** (3 / 2)
        if denominator != 0:
            curvature = numerator / denominator
            curvatures.append(curvature)

    # 计算平均曲率
    if len(curvatures) > 0:
        average_curvature = np.mean(curvatures)
    else:
        average_curvature = 0

    return average_curvature
