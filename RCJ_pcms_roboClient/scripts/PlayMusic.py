#!/usr/bin/env python3

import rospy
from audioplayer import AudioPlayer
from std_msgs.msg import Empty
from rospkg import RosPack


def callback(empty):
    global player
    player.play()


def pause_callback(empty):
    global player
    player.pause()


rospy.init_node('birthday_song')

base = RosPack().get_path('home_robotics')
player = AudioPlayer(f'{base}/music/birthday_song.mp3')

rospy.Subscriber(
    '~play',
    Empty,
    callback,
    queue_size=1
)

rospy.Subscriber(
    '~pause',
    Empty,
    pause_callback,
    queue_size=1
)

rospy.spin()
