import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, coords, frontiers, robot_belief, target_position, costmap: np.ndarray = None):
        """
        coords: (x, y)
        """
        self.coords = coords
        self.observable_frontiers = []
        self.sensor_range = 80
        self.target_position = target_position
        self.direction_vector = self.get_direction_vector()
        self.cost = 0
        if costmap is not None:
            self.cost = costmap[int(self.coords[1]), int(self.coords[0])]
        self.initialize_observable_frontiers(frontiers, robot_belief)
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.utiliy = 1
            self.zero_utility_node = False

    def initialize_observable_frontiers(self, frontiers, robot_belief):
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < self.sensor_range - 10]
        for point in frontiers_in_range:
            collision = self.check_collision(self.coords, point, robot_belief)
            if not collision:
                self.observable_frontiers.append(point)

    def get_direction_vector(self):
        dx = self.target_position[0] - self.coords[0]
        dy = self.target_position[1] - self.coords[1]
        mag = (dx**2 + dy**2) ** 0.5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        if mag > 80:
            mag = 80
        return [dx, dy, mag]

    def get_node_utility(self):
        return len(self.observable_frontiers)

    def get_node_cost(self):
        return self.cost

    def update_observable_frontiers(self, observed_frontiers, new_frontiers, robot_belief):
        """
        robot_belief: current lidar scan
        observed_frontiers: observed frontiers for current robot's location
        new_frontiers: new frontiers for current robot's location
        """
        # print(f"observed_frontiers.shape: {observed_frontiers.shape}")
        # if observed_frontiers != []:
        if observed_frontiers.shape[0] != 0:
            observed_index = []
            for i, point in enumerate(self.observable_frontiers):
                if point[0] + point[1] * 1j in observed_frontiers[:, 0] + observed_frontiers[:, 1] * 1j:
                    observed_index.append(i)
            for index in reversed(observed_index):
                self.observable_frontiers.pop(index)

        # if new_frontiers != []:
        if new_frontiers.shape[0] != 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.sensor_range - 10]
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_belief)
                if not collision:
                    self.observable_frontiers.append(point)
        self.utility = self.get_node_utility()
        if self.utility < 5:
            self.utility = 0
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self):
        self.observable_frontiers = []
        self.utility = 0
        self.zero_utility_node = True

    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision
