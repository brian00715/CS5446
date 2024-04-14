"""
 # @ Author: Kenneth Simon
 # @ Email: smkk00715@gmail.com
 # @ Create Time: 2024-04-13 19:49:37
 # @ Modified time: 2024-04-15 01:35:59
 # @ Description:
 """

import time

import actionlib
import numpy as np
import rospy
import tf.transformations as tft
import torch
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalID, GoalStatus
from gazebo import Gazebo
from geometry_msgs.msg import PoseStamped, Twist
from gymnasium import Env
from move_base_msgs.msg import (MoveBaseAction, MoveBaseActionFeedback,
                                MoveBaseActionResult, MoveBaseFeedback,
                                MoveBaseGoal, MoveBaseResult)
from nav_msgs.msg import Odometry
from PIL import Image
from sensor_msgs.msg import Image, LaserScan
from std_srvs.srv import Empty
from torchvision import transforms

from utils import euclidian_dist_se2, pose2ndarray_se2, quat2yaw, yaw2quat_ros


class GazeboEnv(Env):
    def __init__(self, rate=30):
        super().__init__()
        self.rate = rate
        self.ros_rate = rospy.Rate(self.rate)

        # states variables
        self.observation_space = None
        self.action_space = None

        self.img_curr = Image()
        self.scan_curr = LaserScan()
        self.scan_curr_dict = None
        self.pose_curr = [0, 0, 0]  # x,y,theta
        self.odom_curr = Odometry()
        self.mb_feedback = MoveBaseFeedback()
        self.mb_goal_id_curr = None
        self.mb_done = None
        self.mb_goal_curr = PoseStamped()

        # ros related
        # subs
        self.rgb_sub = rospy.Subscriber("/front/image_raw", Image, self.rgb_callback)
        self.odom_sub = rospy.Subscriber("/gazebo/ground_truth/state", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/front/scan", LaserScan, self.scan_callback)
        # pubs
        self.pose_goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # srvs
        self.reset_world_srv = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.reset_simulation_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        # actions
        self.move_base_act = SimpleActionClient("move_base", MoveBaseAction)
        self.move_base_act.wait_for_server(rospy.Duration(5))
        rospy.loginfo("Connected to move_base server")

        # context embedding
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True).to(self.device)
        # self.resnet.eval()
        # self.preprocess = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )

    def move_base_status_callback(self, msg: MoveBaseFeedback):
        self.mb_feedback = msg
        # rospy.loginfo(f"move_base_feedback: {pose2ndarray_se2(msg.base_position.pose)}")
        yaw_goal = quat2yaw(self.mb_goal_curr.pose.orientation)
        yaw_curr = quat2yaw(msg.base_position.pose.orientation)
        if euclidian_dist_se2(msg.base_position.pose, self.mb_goal_curr.pose) < 0.15 and abs(yaw_goal - yaw_curr) < 0.2:
            self.mb_done = GoalStatus.SUCCEEDED
            self.move_base_act.cancel_goal()

    # NOTE: oftentimes, the move_base action can't set the done status successfully, so we need to use the feedback to determine the done status.
    # def move_base_done_callback(self, status: GoalStatus, result: MoveBaseResult):
    #     self.mb_done = status
    # rospy.loginfo(f"move_base_done_result: {result}, type:{type(result)}")
    # rospy.loginfo(f"move_base_done_status: {status}, type:{type(status)}")

    def scan_callback(self, msg):
        self.scan_curr = msg
        scan = {
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "ranges": np.array(msg.ranges),
        }
        self.scan_curr_dict = scan

    def rgb_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.img_curr = img

    def odom_callback(self, msg):
        self.odom_curr = msg
        self.pose_curr = pose2ndarray_se2(msg.pose.pose)

    def get_obs(self) -> torch.Tensor:
        # pil = Image.fromarray(self.img_curr)
        # input_tensor = self.preprocess(pil)
        # input_batch = input_tensor.unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     predict = self.resnet(input_batch)
        # return predict
        state = {
            "img": self.img_curr,
            "scan": self.scan_curr_dict,
            "pose": self.pose_curr,
        }
        return state

    def apply_action(self, action: np.ndarray):
        """

        action: [x, y, theta]
        """
        # msg = PoseStamped()
        # msg.pose.position.x = action[0]
        # msg.pose.position.y = action[1]
        # yaw = tft.quaternion_from_euler(0, 0, action[2])
        # msg.pose.orientation = yaw2quat_ros(yaw)
        # self.pose_goal_pub(pub_msg=msg)

        self.gold_id_curr = np.random.randint(0, 1000)
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.stamp = rospy.Time.now()
        move_base_goal.target_pose.header.frame_id = "map"
        self.mb_goal_curr = PoseStamped()
        self.mb_goal_curr.header.stamp = rospy.Time.now()
        self.mb_goal_curr.header.frame_id = "map"
        self.mb_goal_curr.pose.position.x = action[0]
        self.mb_goal_curr.pose.position.y = action[1]
        self.mb_goal_curr.pose.orientation = yaw2quat_ros(action[2])
        move_base_goal.target_pose.pose = self.mb_goal_curr.pose
        self.move_base_act.send_goal(move_base_goal, feedback_cb=self.move_base_status_callback)

    def step(self, action):
        """

        action: [x, y, theta]
        """
        self.apply_action(action)
        rospy.loginfo(f"action: {action} is applied.")
        if 0:
            rst = self.move_base_act.wait_for_result()
            if rst:
                rospy.loginfo("move_base action is done.")
            else:
                rospy.loginfo("move_base action is not done.")
        if 1:
            while not self.mb_done == GoalStatus.SUCCEEDED:
                rospy.loginfo_throttle(2, "waiting for move_base action to be done.")
                self.ros_rate.sleep()
            rospy.loginfo("move_base action is done.")
        obs = self.get_obs()
        return obs

    def reset(self):
        try:
            self.move_base_act.cancel_all_goals()
            self.reset_world_srv()
            return True
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False

    def render(self):
        pass


if __name__ == "__main__":
    rospy.init_node("agent")
    env = GazeboEnv()
    # obs = env.step([7, 3.3, 0])
    obs = env.step([1, 0, 0])
    # obs = env.step([0, 0, 0])
    print(f"obs: {obs['img'].shape, obs['scan']['ranges'].shape, obs['pose']}")

    res = env.reset()
    print(f"res: {res}")
