import time

import rospy
import tf.transformations as tft
import torch
from gazebo_agent import GazeboAgent
from geometry_msgs.msg import PoseStamped
from torchvision import transforms

from utils import yaw2quat_ros
from PIL import Image


class PolictAgent(GazeboAgent):
    def __init__(self, rate=30):
        super().__init__()
        self.rate = rate

        # context embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True).to(self.device)
        self.resnet.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_obs(self) -> torch.Tensor:
        pil = Image.fromarray(self.img_curr)
        input_tensor = self.preprocess(pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            predict = self.resnet(input_batch)
        return predict

    def apply_action(self, action):
        msg = PoseStamped()
        msg.pose.position.x = action[0]
        msg.pose.position.y = action[1]
        yaw = tft.quaternion_from_euler(0, 0, action[2])
        msg.pose.orientation = yaw2quat_ros(yaw)
        self.pose_goal_pub(pub_msg=msg)

    def rollout(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            # obs = self.get_obs()
            # action = self.policy(obs)
            # self.apply_action(action)
            rospy.loginfo(f"{self.get_obs().shape}")
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("agent")
    agent = PolictAgent()
    agent.rollout()
