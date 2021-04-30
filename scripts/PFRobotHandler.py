#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2020 rootadminWalker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from home_robot_msgs.msg import PFRobotData, PFWaypoint
from sensor_msgs.msg import Image

from core.tools import PIDController


class PFRobotHandler:
    H = 480
    W = 640
    CENTROID = (W // 2, H // 2)

    FORWARD_KP = 1 / 800
    FORWARD_KD = 1 / 1200

    TURN_KP = -(1 / 200)
    TURN_KD = 1 / 300

    # SMOOTH_CONTROL_KP = 0
    # SMOOTH_CONTROL_KD = 0
    SMOOTH_CONTROL_KP = 0.1
    SMOOTH_CONTROL_KD = 0.15

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
        self.fake_waypoint = []

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
        self.fake_waypoint_pub = rospy.Publisher(
            '~fake_waypoint',
            PFWaypoint,
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

        distance = self.__avoid_zeropoints(centroid, self.depth_image)
        x, y = centroid
        centroid_x = PFRobotHandler.CENTROID[0]

        forward_error = distance - PFRobotHandler.TARGET_DIST
        target_forward_speed = self.forward_controller.update(forward_error)
        new_speed = self.__smooth_acceleration(self.forward_speed, target_forward_speed)

        # target_speed_error = target_forward_speed - self.forward_speed
        # smooth_speed = self.smooth_controller.update(target_speed_error)
        # print(target_speed_error)

        rospy.loginfo(distance)
        if abs(new_speed) <= PFRobotHandler.FORWARD_SPEED_LIMIT:
            self.forward_speed = new_speed

        turn_error = x - centroid_x
        target_turn_speed = self.turn_controller.update(turn_error)
        self.turn_speed = target_turn_speed

        self.fake_waypoint = [x, y, distance]

        rospy.loginfo(f'Forward_speed: {self.forward_speed}, turn_speed: {self.turn_speed}')

    def __smooth_acceleration(self, current_speed, target_speed):
        target_speed_error = target_speed - current_speed
        smooth_speed = self.smooth_controller.update(target_speed_error)
        return current_speed + smooth_speed

    @staticmethod
    def __avoid_zeropoints(point, depth_image):
        x, y = point
        for each in range(1, depth_image.shape[0]):
            up = y - each
            down = y + each
            left_x = x - each
            right_x = x + each

            top = depth_image[up:up + 1, left_x:right_x + 1]
            left = depth_image[up:up + 1, left_x:left_x + 1]
            bottom = depth_image[down:down + 1, left_x:left_x + 1]
            right = depth_image[up:down + 1, right_x:right_x + 1]

            for block in [top, left, bottom, right]:
                nonzero = block[np.nonzero(block)]
                if nonzero.shape[0] > 0:
                    distance = nonzero[0]
                    return distance

        return 0


if __name__ == '__main__':
    rospy.init_node('PFRHandler')
    node = PFRobotHandler()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        fake_waypoint = PFWaypoint()
        if node.rgb_image is None:
            continue

        node.rgb_publisher.publish(node.rgb_image)

        twist = Twist()
        forward_speed = node.forward_speed
        turn_speed = node.turn_speed

        twist.linear.x = forward_speed
        twist.angular.z = turn_speed
        node.twist_publisher.publish(twist)

        if len(node.fake_waypoint) > 0:
            fake_waypoint.x = node.fake_waypoint[0]
            fake_waypoint.y = node.fake_waypoint[1]
            fake_waypoint.z = node.fake_waypoint[2]
            node.fake_waypoint_pub.publish(fake_waypoint)

        rate.sleep()
