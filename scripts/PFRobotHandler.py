#!/usr/bin/env python3
import rospy
from core.tools import PIDController
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from home_robot_msgs.msg import PFRobotData
from sensor_msgs.msg import Image


class PFRobotHandler:
    H = 480
    W = 640
    CENTROID = (W // 2, H // 2)

    FORWARD_KP = 1 / 600
    FORWARD_KD = 1 / 800

    TURN_KP = -(1 / 120)
    TURN_KD = 1 / 200

    # SMOOTH_CONTROL_KP = 1 / 5
    # SMOOTH_CONTROL_KP = 1 / 5
    SMOOTH_CONTROL_KP = 0.05
    SMOOTH_CONTROL_KD = 0.01

    FORWARD_SPEED_LIMIT = 1.1

    TARGET_DIST = 1040

    def __init__(self):
        self.forward_speed = 0
        self.turn_speed = 0

        self.forward_controller = PIDController(
            PFRobotHandler.FORWARD_KP, 0, PFRobotHandler.FORWARD_KD
        )
        self.turn_controller = PIDController(
            PFRobotHandler.TURN_KP, 0, PFRobotHandler.TURN_KD
        )
        self.smooth_controller = PIDController(
            PFRobotHandler.SMOOTH_CONTROL_KP, 0, PFRobotHandler.SMOOTH_CONTROL_KD
        )

        self.bridge = CvBridge()

        self.rgb_image = self.tmp_depth_image = self.depth_image = None

        rospy.Subscriber(
            '~pf_data',
            PFRobotData,
            self.info_callback,
            queue_size=1
        )
        rospy.Subscriber(
            '/camera/rgb/image_raw',
            Image,
            self.rgb_callback,
            queue_size=1
        )
        rospy.Subscriber(
            '/camera/depth/image_raw',
            Image,
            self.depth_callback,
            queue_size=1
        )
        self.rgb_publisher = rospy.Publisher(
            '~rgb/image_raw',
            Image,
            queue_size=1
        )
        self.twist_publisher = rospy.Publisher(
            '/mobile_base/commands/velocity',
            Twist,
            queue_size=1
        )

    def rgb_callback(self, rgb: Image):
        self.rgb_image = rgb

        PFRobotHandler.H = self.rgb_image.height
        PFRobotHandler.W = self.rgb_image.width
        PFRobotHandler.CENTROID = (PFRobotHandler.W // 2, PFRobotHandler.H // 2)

    def depth_callback(self, depth: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth)

    def info_callback(self, msg: PFRobotData):
        centroid = msg.follow_point
        if centroid == (-1, -1):
            self.turn_speed = 0
            self.forward_speed = self.__smooth_acceleration(self.forward_speed, 0)
            return

        distance = self.depth_image[centroid[::-1]]
        x = centroid[0]
        centroid_x = PFRobotHandler.CENTROID[0]

        forward_error = distance - PFRobotHandler.TARGET_DIST
        target_forward_speed = self.forward_controller.update(forward_error)
        smooth_speed = self.__smooth_acceleration(self.forward_speed, target_forward_speed)

        # target_speed_error = target_forward_speed - self.forward_speed
        # smooth_speed = self.smooth_controller.update(target_speed_error)
        # print(target_speed_error)

        self.forward_speed = min(smooth_speed, PFRobotHandler.FORWARD_SPEED_LIMIT)

        turn_error = x - centroid_x
        target_turn_speed = self.turn_controller.update(turn_error)
        self.turn_speed = target_turn_speed

        rospy.loginfo(f'Forward_speed: {self.forward_speed}, turn_speed: {self.turn_speed}')

    def __smooth_acceleration(self, current_speed, target_speed):
        target_speed_error = target_speed - current_speed
        smooth_speed = self.smooth_controller.update(target_speed_error)
        return current_speed + smooth_speed


if __name__ == '__main__':
    rospy.init_node('PFRHandler')
    node = PFRobotHandler()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        if node.rgb_image is None:
            continue

        node.rgb_publisher.publish(node.rgb_image)

        twist = Twist()
        forward_speed = node.forward_speed
        turn_speed = node.turn_speed

        twist.linear.x = forward_speed
        twist.angular.z = turn_speed
        node.twist_publisher.publish(twist)

        rate.sleep()
