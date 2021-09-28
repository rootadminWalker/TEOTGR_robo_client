#!/usr/bin/env python3

import rospy
from actionlib_msgs.msg import GoalStatusArray


def status_callback(data):
    if len(data.status_list) > 0:
        status_code = data.status_list[-1].status
        rospy.set_param('~status_code', int(status_code))


rospy.init_node('status_monitor')
rospy.Subscriber(
    "/move_base/status",
    GoalStatusArray,
    status_callback,
    queue_size=1
)
rospy.set_param('~status_code', 0)
rospy.spin()
