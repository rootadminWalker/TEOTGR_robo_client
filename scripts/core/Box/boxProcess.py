#!/usr/bin/env python3

"""
MIT License

Copyright (c) 2019 rootadminWalker

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

from home_robot_msgs.msg import ObjectBox

from imutils.object_detection import non_max_suppression
from math import sqrt
from typing import List
import cv2 as cv
import numpy as np
import dlib


class BBox:
    """
    BBox is a helper type for some convenience methods and calculations
    1.1 Call
    method 1
    box = BBox({
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    }, padding=(top, side), shape=image_shape)

    padding (Optional, tuple): This create a same ratio but bigger box.
    shape (Optional, tuple): Use with padding to avoid override

    method 2
    box = posToBBox([[x1, y1, x2, y2], [x1, y1, x2, y2]], padding=(top, side), shape=image_shape)

    1.2 Convenience variables

    Get centroid by
    --box.centroid

    Area by
    --box.area

    x1, y1, x2, y2 by
    --box.x1, box.y1, box.x2, box.y2

    Padding-box by
    --box.padding_box

    Height and width by
    --box.height, box.width

    1.3 Convenience methods
    --box.is_inBox(box2)

    Desc:
    Check if box is in box2

    Args:
    box2 (Required, BBox)

    --box.calc_distance_between_boxes(box2)

    Desc:
    Check distance between self and box2

    Args:
    box2 (Required, BBox)

    --box.draw(image, color, thickness)

    Desc:
    Draw itself to a OpenCV image

    Args:
    image (Required, np.array): A OpenCV image
    color (Required, tuple): The color of the rectangle
    thickness (Required, tuple): Thickness of rectangle

    --box.draw_centroid(image, color, thickness)

    Desc:
    Draws itself's centroid to an OpenCV image

    Args:
    image (Required, np.array): A OpenCV image
    color (Required, tuple): The color of the centroid
    thickness (Required, tuple): Thickness of centroid

    --box.as_np_array()

    Desc:
    Return itself as np array coordinates. format: [x1, y1, x2, y2]

    """

    def __init__(self, coordinates, label='', model='', score=0.0, padding=None, shape=None):
        self.label = label
        self.coordinates = coordinates
        self.padding = padding
        self.score = score
        self.model = model

        for key, value in self.coordinates.items():
            self.coordinates[key] = int(value)

        self.x1 = self.coordinates['x1']
        self.y1 = self.coordinates['y1']
        self.x2 = self.coordinates['x2']
        self.y2 = self.coordinates['y2']

        if shape is not None and padding is not None:
            self.padding_box = self.__calc_padding(shape)

        self.area = self.__calc_area()
        self.centroid = self.__calc_centroid()

        self.corresponds = {
            'x1': ['width', 'x2'],
            'y1': ['height', 'y2'],
            'x2': ['width', 'x1'],
            'y2': ['height', 'y1'],
        }

        self.height = self.y2 - self.y1
        self.width = self.x2 - self.x1

        self.serialize_msg = ObjectBox()

    def __repr__(self):
        return '{} at {}, label={}, pos=(x1={}, y1={}, x2={}, y2={}), centroid={}, area={}>'.format(
            self.__class__, hex(id(self)),
            self.label,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.centroid,
            self.area
        )

    def __calc_centroid(self):
        return int(((self.x2 - self.x1) / 2) + self.x1), \
               int(((self.y2 - self.y1) / 2) + self.y1)

    def __calc_area(self) -> int:
        self.area = (self.y2 - self.y1) * (self.x2 - self.x1)
        return self.area

    def __calc_padding(self, shape):
        px1 = self.x1 - self.padding[0]
        py1 = self.y1 - self.padding[1]
        px2 = self.x2 + self.padding[0]
        py2 = self.y2 + self.padding[1]

        self.padding_box = {
            'x1': px1 if px1 > 0 else 0,
            'y1': py1 if py1 > 0 else 0,
            'x2': px2 if px2 < shape[1] else shape[1],
            'y2': py2 if py2 < shape[0] else shape[0]
        }
        return BBox(self.padding_box)

    def is_inBox(self, box2):
        box2_pos = box2.coordinates
        isX1 = self.coordinates['x1'] - box2_pos['x1'] >= 0
        isY1 = self.coordinates['y1'] - box2_pos['y1'] >= 0
        isX2 = self.coordinates['x2'] - box2_pos['x2'] <= 0
        isY2 = self.coordinates['y2'] - box2_pos['y2'] <= 0

        return isX1 and isY1 and isX2 and isY2

    def calc_distance_between_boxes(self, box2: 'BBox'):
        x_distance = self.centroid[0] - box2.centroid[0]
        y_distance = self.centroid[1] - box2.centroid[1]
        return sqrt((x_distance ** 2) + (y_distance ** 2))

    def draw(self, image, color=(255, 32, 255), thickness=5):
        cv.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), color, thickness)

    def draw_centroid(self, image, color, thickness):
        cv.circle(image, self.centroid, 5, color, thickness)

    def putText_at_top(self, image, text, color=(255, 32, 255), thickness=2, font_scale=1):
        text_origin = (self.x1, self.y1 - thickness - font_scale)
        cv.rectangle(image, (self.x1, text_origin[1] - 30), (self.x2, self.y1), color, thickness=-1)
        cv.putText(
            image,
            text,
            text_origin,
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 0, 0), thickness,
            cv.LINE_AA
        )

    def as_np_array(self) -> np.array:
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def as_dlib_rectangle(self) -> dlib.rectangle:
        return dlib.rectangle(self.x1, self.y1, self.x2, self.y2)

    def as_list(self) -> list:
        return [self.x1, self.y1, self.x2, self.y2]

    def divide_part_of_box(self, option, value):
        width_or_height = self.corresponds[option][0]
        correspond_pos = self.corresponds[option][1]

        self.__dict__[width_or_height] *= value
        result_pos = int(self.__dict__[width_or_height] - self.__dict__[correspond_pos])

        self.__dict__[option] = result_pos if result_pos > 0 else -result_pos

    def crop_box_from_image(self, image):
        image = image[self.y1:self.y2, self.x1:self.x2, :].copy()
        return image

    def serialize_ros(self):
        self.serialize_msg.x1 = self.x1
        self.serialize_msg.y1 = self.y1
        self.serialize_msg.x2 = self.x2
        self.serialize_msg.y2 = self.y2

        self.serialize_msg.model = self.model
        self.serialize_msg.score = self.score
        self.serialize_msg.label = self.label

        return self.serialize_msg


def filterBoxes(out_boxes, min_area):
    out_boxes = filter(lambda x: x.area >= min_area, out_boxes)
    return list(out_boxes)


def posToBBox(out_boxes, labels=None, padding=None, shape=None):
    after_bboxing = []
    for idx, (x1, y1, x2, y2) in enumerate(out_boxes):
        after_bboxing.append(BBox(
            {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            },
            labels[idx] if labels is not None else labels,
            padding=padding, shape=shape
        ))

    return after_bboxing


def dlibToBBox(out_boxes: List[dlib.rectangle], padding=None, shape=None):
    after_bboxing = []
    for dlib_box in out_boxes:
        after_bboxing.extend(
            posToBBox([
                dlib_box.left(),
                dlib_box.top(),
                dlib_box.right(),
                dlib_box.bottom()
            ], padding=padding, shape=shape)
        )

    return after_bboxing


def BBoxToPos(out_boxes):
    for box in out_boxes:
        yield box.as_np_array()


def do_nms(boxes, use_bbox=True, padding=None, shape=None):
    processed_box = non_max_suppression(boxes)

    if use_bbox:
        return posToBBox(processed_box, padding, shape)

    return processed_box
