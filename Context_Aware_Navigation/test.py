import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_img_bin_rgb(img, low_r, high_r, low_g, high_g, low_b, high_b):
    _, binary_red = cv2.threshold(img[:, :, 0], low_r, high_r, cv2.THRESH_BINARY)
    _, binary_green = cv2.threshold(img[:, :, 1], low_g, high_g, cv2.THRESH_BINARY)
    _, binary_blue = cv2.threshold(img[:, :, 2], low_b, high_b, cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_or(cv2.bitwise_or(binary_red, binary_green), binary_blue)
    return img_bin


def get_img_bin_hsv(img, low_h, high_h, low_s, high_s, low_v, high_v):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([low_h, low_s, low_v])
    upper_bound = np.array([high_h, high_s, high_v])
    img_bin = cv2.inRange(img_hsv, lower_bound, upper_bound)
    return img_bin


def img_set_color_to(img, colors, mask_value=255):
    colors = np.array(colors)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            color = img[y, x]
            for target_color in colors:
                if color[0] == target_color[2] and color[1] == target_color[1] and color[2] == target_color[0]:
                    mask[y, x] = mask_value
    return mask


def calculate_cost_map(map_data: cv2.Mat, obstacle_cost=100, inflation_radius=3, inflation_scale=1.0):
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


map = cv2.imread("/home/simon/Research/CS5446/Context_Aware_Navigation/DungeonMaps/pp/test/1.png")
inflation_radius = 5
inflation_scale = 0.2
costmap = calculate_cost_map(map, inflation_radius=inflation_radius, inflation_scale=inflation_scale)
cv2.imwrite(f"tmp/1_costmap_{inflation_radius}_{inflation_scale}.png", costmap)
