#!/usr/bin/env python3

import numpy as np
import rospy
from home_robot_msgs.msg import ObjectBoxes
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
from std_msgs.msg import String


class ManipulatorGrab:
    MANI_SRV_NAME = '/goal_task_space_path_position_only'

    MANI_HEIGHT_ERR = 40
    MANI_FRONT_ERR = 10

    WATCH_TIMEOUT = 3

    FOV_H = 60
    FOV_V = 49.5

    CAMERA_ANGLE = -20

    def __init__(self):
        rospy.init_node('manipulator_grab')

        self.bottle_boxes = []

        rospy.wait_for_service(ManipulatorGrab.MANI_SRV_NAME)
        self.service = rospy.ServiceProxy(ManipulatorGrab.MANI_SRV_NAME, SetKinematicsPose)

        self.speaker_pub = rospy.Publisher(
            '/speaker/say',
            String,
            queue_size=1
        )

        rospy.Subscriber(
            '/YD/boxes',
            ObjectBoxes,
            self.box_callback,
            queue_size=1
        )
        self.main()

    def box_callback(self, boxes: ObjectBoxes):
        detection_boxes = boxes.boxes
        self.bottle_boxes = list(filter(lambda x: x.label.strip() == 'bottle', detection_boxes))

    def move_to(self, x, y, z, t):
        try:
            req = SetKinematicsPoseRequest
            req.end_effector_name = 'gripper'
            req.kinematics_pose.pose.position.x = x
            req.kinematics_pose.pose.position.y = y
            req.kinematics_pose.pose.position.z = z
            req.path_time = t

            resp = self.service(req)
            return resp
        except Exception as e:
            rospy.loginfo(e)
            return False

    @staticmethod
    def angle_2_radian(angle):
        return (np.pi * angle) / 180

    def convert_waypoint_to_real(self, x, y, z):
        rad_h = self.angle_2_radian(ManipulatorGrab.FOV_H / 2)
        rad_v = self.angle_2_radian(ManipulatorGrab.FOV_V / 2)
        rad_cam_angle = self.angle_2_radian(ManipulatorGrab.CAMERA_ANGLE)
        real_w = 2 * z * np.tan(rad_h)
        real_h = 2 * z * np.tan(rad_v)

        # Real x
        real_x = real_w * x / ManipulatorGrab.W
        real_x -= (real_w / 2)

        # Real y
        reality_y = real_h * y / ManipulatorGrab.H
        FE = z * np.sin(rad_cam_angle)
        GF = (0.5 * real_h - reality_y) * np.cos(rad_cam_angle)
        real_y = FE + GF

        # Real z
        OD = z * np.cos(rad_cam_angle)
        ED = GF * np.tan(rad_cam_angle)
        real_z = OD - ED

        return real_x, real_y, real_z

    def main(self):
        while not rospy.is_shutdown():
            start_time = rospy.get_rostime() + rospy.Duration(3)
            while rospy.get_rostime() - start_time <= 0:
                if len(self.bottle_boxes) != 1:
                    start_time = rospy.get_rostime() + rospy.Duration(3)
