#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import numpy as np
from utils import pose2ndarray_se2, yaw2quat_ros
import cv2


class Gazebo:
    def __init__(self):

        # status variables
        self.img_curr = None
        self.pose_curr = [0, 0, 0]  # x,y,theta
        self.odom_curr = None

        # ros related
        # subs
        self.rgb_sub = rospy.Subscriber("/front/image_raw", Image, self.rgb_callback)
        self.odom_sub = rospy.Subscriber("/gazebo/ground_truth/state", Odometry, self.odom_callback)
        # pubs
        self.pose_goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

        # rospy.spin()

    def rgb_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.img_curr = img
        # rospy.loginfo(f"Received image: {self.img_curr.shape}")

    def odom_callback(self, msg):
        self.odom_curr = msg
        self.pose_curr = pose2ndarray_se2(msg.pose.pose)


if __name__ == "__main__":
    rospy.init_node("gazebo_agent")
    agent = GazeboEnv()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if agent.img_curr is None:
            continue
        cv2.imshow("image", agent.img_curr)
        cv2.waitKey(1)
        rate.sleep()
