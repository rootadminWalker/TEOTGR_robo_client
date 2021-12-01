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
from home_robot_msgs.msg import PFRobotData, PFWaypoint
from sensor_msgs.msg import Image

from core.Nodes import Node
from core.tools import PIDController, Chassis


class PFRobotHandler(Node):
    H = 480
    W = 640
    CENTROID = (W // 2, H // 2)

    FORWARD_KP = 1 / 800
    FORWARD_KD = 1 / 1200

    TURN_KP = -(1 / 200)
    TURN_KD = 1 / 300

    # SMOOTH_CONTROL_KP = 0
    # SMOOTH_CONTROL_KD = 0
    SMOOTH_CONTROL_KP = 0.15
    SMOOTH_CONTROL_KD = 0.2

    DANGER_SMOOTH_CONTROL_KP = 0.25
    DANGER_SMOOTH_CONTROL_KD = 0.3

    FORWARD_SPEED_LIMIT = 1.
    MAXIMUM_ACCELERATION = 0.2

    TARGET_NORMAL_DIST = 1040
    TARGET_SEARCHING_DIST = 780
    DANGER_DIST = 690
    MAX_DIST = 3000

    def __init__(self):
        super(PFRobotHandler, self).__init__('PFRHandler', anonymous=False)
        self.forward_speed = 0
        self.turn_speed = 0

        self.last_forward_speed = self.last_turn_speed = 0
        self.last_dist = 0

        self.forward_controller = PIDController(
            PFRobotHandler.FORWARD_KP, 0, PFRobotHandler.FORWARD_KD
        )
        self.turn_controller = PIDController(
            PFRobotHandler.TURN_KP, 0, PFRobotHandler.TURN_KD
        )

        # self.smooth_controller = PIDController(
        #     PFRobotHandler.SMOOTH_CONTROL_KP, 0, PFRobotHandler.SMOOTH_CONTROL_KD
        # )
        # self.danger_smooth_controller = PIDController(
        #     PFRobotHandler.DANGER_SMOOTH_CONTROL_KP, 0, PFRobotHandler.DANGER_SMOOTH_CONTROL_KD,
        # )

        self.smooth_controller = SmoothAcceleration(
            PFRobotHandler.SMOOTH_CONTROL_KP, 0, PFRobotHandler.SMOOTH_CONTROL_KD, PFRobotHandler.MAXIMUM_ACCELERATION
        )
        self.confirm_lost_smooth_controller = SmoothAcceleration(
            PFRobotHandler.SMOOTH_CONTROL_KP, 0, PFRobotHandler.SMOOTH_CONTROL_KD,
            PFRobotHandler.CONFIRM_LOST_ACCELERATION
        )

        self.bridge = CvBridge()

        self.chassis = Chassis(cmd_topic='/mobile_base/commands/velocity')

        self.rgb_image = self.tmp_depth_image = self.depth_image = None
        self.fake_waypoint = []

        self.centroid = (-1, -1)

        rospy.Subscriber(
            '/PFRHandler/pf_data',
            PFRobotData,
            self.info_callback,
            queue_size=1
        )
        rospy.Subscriber(
            '/top_camera/rgb/image_raw',
            Image,
            self.rgb_callback,
            queue_size=1
        )
        rospy.Subscriber(
            '/top_camera/depth/image_raw',
            Image,
            self.depth_callback,
            queue_size=1
        )
        self.rgb_publisher = rospy.Publisher(
            '~rgb/image_raw',
            Image,
            queue_size=1
        )
        # self.twist_publisher = rospy.Publisher(
        #     '/mobile_base/commands/velocity',
        #     Twist,
        #     queue_size=1
        # )
        self.fake_waypoint_pub = rospy.Publisher(
            '~fake_waypoint',
            PFWaypoint,
            queue_size=1
        )
        self.main()

    def rgb_callback(self, rgb: Image):
        self.rgb_image = rgb

        PFRobotHandler.H = self.rgb_image.height
        PFRobotHandler.W = self.rgb_image.width
        PFRobotHandler.CENTROID = (PFRobotHandler.W // 2, PFRobotHandler.H // 2)

    def depth_callback(self, depth: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth)

    def info_callback(self, msg: ObjectBox):
        bbox = BBox.from_ObjectBox(msg)
        self.centroid = bbox.centroid

    def __smooth_acceleration(self, current_speed, target_speed, smooth_controller):
        target_speed_error = target_speed - current_speed
        smooth_speed = smooth_controller.update(target_speed_error)
        if abs(smooth_speed) > PFRobotHandler.MAXIMUM_ACCELERATION:
            if smooth_speed < 0:
                smooth_speed = -PFRobotHandler.MAXIMUM_ACCELERATION
            else:
                smooth_speed = PFRobotHandler.MAXIMUM_ACCELERATION

        rospy.loginfo(smooth_speed)
        return current_speed + smooth_speed

    @staticmethod
    def __avoid_zeropoints(point, depth_image, limit=None):
        if limit is None:
            limit = depth_image.shape[0]

        x, y = point
        for each in range(1, limit):
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

        return -1

    @staticmethod
    def get_follower_state():
        return rospy.get_param('/person_follower/state')

    def reset(self):
        pass

    def main(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            fake_waypoint = PFWaypoint()
            state = self.get_follower_state()

            if state in ['NOT_INITIALIZED', 'LOST', 'CONFIRM_REIDENTIFIED'] or self.depth_image is None:
                self.turn_speed = 0
                self.forward_speed = self.__smooth_acceleration(self.forward_speed, 0, self.smooth_controller)

            else:
                forward_smooth_controller = self.smooth_controller

                origin_distance = self.__avoid_zeropoints(self.centroid, self.depth_image, limit=30)
                target_dist = PFRobotHandler.TARGET_NORMAL_DIST

                distance = self.last_dist if origin_distance == -1 else origin_distance

                x, y = self.centroid
                centroid_x = PFRobotHandler.CENTROID[0]

                forward_error = distance - target_dist
                target_forward_speed = self.forward_controller.update(forward_error)

                turn_error = x - centroid_x
                target_turn_speed = self.turn_controller.update(turn_error)
                final_turn_speed = target_turn_speed

                if distance <= 3000:
                    if state == 'CONFIRM_LOST':
                        if self.last_forward_speed - target_forward_speed < 0 or target_forward_speed < 0:
                            forward_smooth_controller = self.confirm_lost_smooth_controller
                        final_turn_speed = forward_stooth_controller.smooth_speed(self.turn_speed, target_turn_speed)
                else:
                    target_forward_speed = 0

                rospy.loginfo(f'dist:{origin_distance}, error:{forward_error}, target: {target_forward_speed}')
                new_speed = forward_smooth_controller.smooth_speed(self.forward_speed, target_forward_speed)

                if abs(new_speed) <= PFRobotHandler.FORWARD_SPEED_LIMIT:
                    self.forward_speed = new_speed

                self.turn_speed = final_turn_speed

                self.fake_waypoint = [x, y, distance]

                rospy.loginfo(f'Forward_speed: {self.forward_speed}, turn_speed: {self.turn_speed}')

                if self.rgb_image is None:
                    continue

                self.rgb_publisher.publish(self.rgb_image)

                self.chassis.move(self.forward_speed, self.turn_speed)

                if len(self.fake_waypoint) > 0:
                    fake_waypoint.x = self.fake_waypoint[0]
                    fake_waypoint.y = self.fake_waypoint[1]
                    fake_waypoint.z = self.fake_waypoint[2]
                    self.fake_waypoint_pub.publish(fake_waypoint)

                self.last_dist = distance
                self.last_forward_speed = self.forward_speed
                self.last_turn_speed = self.turn_speed
            rate.sleep()


if __name__ == '__main__':
    node = PFRobotHandler()
