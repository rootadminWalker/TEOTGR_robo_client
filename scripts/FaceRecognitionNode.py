#!/usr/bin/env python3
import os
import threading
from typing import List

import cv2 as cv
import dlib
import numpy as np
import rospy
from PIL import Image, ImageFont, ImageDraw
from home_robot_msgs.msg import ObjectBoxes, ObjectBox
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from core.Dtypes.FaceProcess import FaceUserManager, FaceUser
from core.Dtypes.boxProcess import dlibToBBox
from core.Nodes.Visions import FaceRecognition
from core.base_classes import Node


class FaceRecognitionNode(Node):
    def __init__(self, name: str = 'node', anonymous: bool = False):
        super(FaceRecognitionNode, self).__init__(name, anonymous)
        self.base = os.path.dirname(os.path.realpath(__file__))

        self._recognizer_path = f'{self.base}/../models/face_recognition/dlib_face_recognition_resnet_model_v1.dat'
        self._shape_predictor_path = f'{self.base}/../models/face_recognition/shape_predictor_68_face_landmarks.dat'

        self.recognizer = dlib.face_recognition_model_v1(self._recognizer_path)
        self.shape_predictor = dlib.shape_predictor(self._shape_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()

        self.face_recognizer = FaceRecognition('face_recognize', self.recognizer, self.shape_predictor)
        self.user_manager = FaceUserManager(f'{self.base}/../models/scientists_recognition/scientists.pickle')

        self.detections = ObjectBoxes()

        self.lock = threading.RLock()

        self.scientist_counter = {}
        self.exist_scientist_list = []
        self.exceed_scientist_count = 10

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

        rospy.set_param('~run', True)
        rospy.set_param('~reset_scientist', False)

        rospy.Subscriber(
            '/camera/rgb/image_raw',
            CompressedImage,
            self.image_callback,
            queue_size=1
        )

        self.detections_pub = rospy.Publisher(
            '~detected_faces',
            ObjectBoxes,
            queue_size=1
        )

        self.speaker_pub = rospy.Publisher(
            '/speaker/text',
            String,
            queue_size=1
        )

    def image_callback(self, image: CompressedImage):
        if rospy.get_param('~reset_scientist'):
            self.scientist_counter = {}
            rospy.set_param('~reset_scientist', False)

        if rospy.get_param('~run'):
            buffer = np.ndarray(shape=(1, len(image.data)),
                          dtype=np.uint8, buffer=image.data)
            frame = cv.imdecode(buffer, cv.IMREAD_COLOR)
            faces: List[dlib.rectangle] = self.face_detector(frame)
            for face in faces:
                scientist: FaceUser = self.face_recognizer.run(
                    serialize=False,
                    image=frame,
                    box=face,
                    user_manager=self.user_manager
                )

                face_box = dlibToBBox([face])[0]
                face_box.draw(frame, color=(32, 255, 0))

                cv.imshow('frame', frame)
                cv.waitKey(16)

                if scientist is None:
                    continue

                scientist_name = scientist.username

                face_box.label = str(self.scientist_id_map[scientist_name])

                if scientist_name not in self.scientist_counter:
                    self.scientist_counter[scientist_name] = 1
                else:
                    self.scientist_counter[scientist_name] += 1

                if self.scientist_counter[scientist_name] > self.exceed_scientist_count:
                    frame = self.draw_text(frame, scientist_name, (face_box.x1, face_box.x2), color=(0, 255, 0))

                    source_img = cv.imencode('.jpg', frame)[1]
                    source_img = CompressedImage(data=np.array(source_img).tostring())

                    serialized_face_box: ObjectBox = face_box.serialize_ros(source_img=source_img)
                    self.detections.boxes.append(serialized_face_box)

            with self.lock:
                try:
                    self.detections_pub.publish(self.detections)
                except rospy.exceptions.ROSSerializationException as e:
                    rospy.logwarn(e)
                    self.detections = ObjectBoxes()
                    return

                self.detections = ObjectBoxes()

    def draw_text(self, image, text, pos, color):
        img_PIL = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        font = ImageFont.truetype(f'{self.base}/../font/simsun.ttc', size=80)
        if not isinstance(text, np.unicode):
            text = text.decode('utf-8')

        draw = ImageDraw.Draw(img_PIL)
        draw.text(pos, text, font=font, fill=color)
        img = cv.cvtColor(np.array(img_PIL), cv.COLOR_RGB2BGR)
        return img

    def reset(self):
        pass


if __name__ == '__main__':
    node = FaceRecognitionNode(name='face_recognition')
    node.spin()
