#!/usr/bin/env python3
from datetime import datetime

import cv2 as cv
import numpy as np
import rospy
from home_robot_msgs.msg import ObjectBoxes
from std_msgs.msg import String

from core.base_classes import Node


class TurtlebotHand(Node):
    def __init__(self, name: str = 'node', anonymous: bool = False):
        super(TurtlebotHand, self).__init__(name, anonymous)
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
        
        self.faces = list()

    def reset(self):
        pass

    def face_recognition_callback(self, faces: ObjectBoxes):
        self.faces = faces.boxes
        for face in self.faces:
            self.speaker_pub.publish(f'你好，{face.label}')
            buffer = np.ndarray(shape=(1, len(face.source_img.data)),
                                dtype=np.uint8, buffer=face.source_img.data)
            source_img = cv.imdecode(buffer, cv.IMREAD_COLOR)
            cv.imwrite(f'/home/mustar/face_pictures/{datetime.now()}.jpg', source_img)
        
    
if __name__ == '__main__':
    node = TurtlebotHand(name='turtlebot_hand')
    node.spin()
