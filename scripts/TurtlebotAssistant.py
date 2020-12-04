#!/usr/bin/env python3
import threading
from datetime import datetime

import rospy
import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist
from home_robot_msgs.msg import ObjectBoxes
from std_msgs.msg import String

from core.base_classes import Node


class TurtlebotAssistant(Node):
    def __init__(self, name: str = 'node', anonymous: bool = False):
        super(TurtlebotAssistant, self).__init__(name, anonymous)

        self.current_target = -1
        self.current_state = 'pending'
        self.move = False

        self.lock = threading.RLock()

        self.seen_scientist = []

        self.scientist_id_map = {
            '钱学森': 1,
            '于敏': 2,
            '茅以升': 3,
            '钟南山': 4,
            '邓稼先': 5,
            '袁隆平': 6,
            '李四光': 7,
            '钱三强': 8,
            '华罗庚': 9
        }

        self.reverse_scientist_id_map = {str(v): k for k, v in self.scientist_id_map.items()}

        self.id_position_map = {
            1: [3.28564278687, 0.643047677701, 0.7175154107],
            2: [3.2847119503, -2.49151134449, -0.69958887],
            3: [4.36792655052, 0.525852279, -0.938577892665],
            4: [4.29586511086, -2.4714828253, 0.947794387005],
            5: [5.42706639887, 0.591687462471, 0.974917728485],
            6: [5.37814945142, -2.65998236867, -0.970003996973],
            7: [6.39873295242, 0.06404144, -0.71701126037],
            8: [6.31093727035, -2.28924701787, 0.585408028038],
            9: [6.464579762, -1.09810222152, -0.10405248014],
        }

        self.origin_position = [2.95693356158, -0.992036307195, 0.0642970216848]

        rospy.Subscriber(
            '/face_recognition/detected_faces',
            ObjectBoxes,
            self.face_recognition_callback,
            queue_size=1
        )

        self.speaker_pub = rospy.Publisher(
            '/speaker/text',
            String,
            queue_size=1
        )

        self.slam_goal_pub = rospy.Publisher(
            '/tb3_nav/goal',
            Twist,
            queue_size=1
        )

        rospy.set_param('~state', self.current_state)

    def face_recognition_callback(self, faces: ObjectBoxes):
        with self.lock:
            faces = faces.boxes
            rospy.loginfo('Get message')
            if self.current_state is 'pending':
                rospy.loginfo(f'State: {self.current_state}')
                if len(faces) > 1 or len(faces) == 0:
                    return

                current_id = faces[0].label
                current_name = self.reverse_scientist_id_map[current_id]
                self.current_target = current_id

                self.seen_scientist.append(current_name)

                self.slam_goal_pub.publish(self.xyz_to_twist(*self.id_position_map[int(current_id)]))

                self.speaker_pub.publish(f'辨識結果為，{current_name}')
                rospy.sleep(0.1)
                self.speaker_pub.publish(f'現在前往目標')

                self.current_state = 'delay'
                self.move = True

            if self.current_state is 'delay':
                rospy.loginfo(f'State: {self.current_state}')
                for face in faces:
                    rospy.loginfo(self.reverse_scientist_id_map[face.label])
                    if face.label == self.current_target:
                        return
                else:
                    rospy.loginfo('in')
                    if self.move:
                        self.current_state = 'moving'
                    else:
                        self.current_state = 'pending'

            if self.current_state is 'moving':
                rospy.loginfo(f'State: {self.current_state}')
                for face in faces:
                    rospy.loginfo(f'State: {self.current_state}, inside loop')
                    if face.label == self.current_target:
                        buffer = np.ndarray(shape=(1, len(face.source_img.data)),
                                                dtype=np.uint8, buffer=face.source_img.data)
                        source_img = cv.imdecode(buffer, cv.IMREAD_COLOR)
                        cv.imwrite(f'/home/mustar/face_pictures/{str(datetime.now())}.jpg', source_img)
                        break
                else:
                    return

                self.slam_goal_pub.publish(self.origin_position)
                self.speaker_pub.publish(f'你好: {self.reverse_scientist_id_map[self.current_target]}')
                rospy.sleep(0.1)
                self.speaker_pub.publish(f'現在回到起始點')
                while rospy.get_param('/tb3_nav/status_code') != 1:
                    rospy.loginfo('Inside waiting loop')
                    continue

                self.speaker_pub.publish('巳回到起始點，請傳遞下一個科學家的樣貌')

                self.current_state = 'delay'
                self.move = False
                if len(self.seen_scientist) >= 9:
                    self.speaker_pub.publish('所有的目標也巳去完，下次見?')
                    rospy.signal_shutdown('Task finished')

    @staticmethod
    def xyz_to_twist(x, y, z):
        new_msg = Twist()
        new_msg.linear.x = x
        new_msg.linear.y = y
        new_msg.angular.z = z
        return new_msg

    def reset(self):
        pass


if __name__ == '__main__':
    node = TurtlebotAssistant(name='turtlebot_assistant')
    rospy.sleep(2)
    node.speaker_pub.publish('你好，我是人工智能木星機器人，請給我科學家的樣貌')
    node.spin()
