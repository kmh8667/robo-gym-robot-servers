#!/usr/bin/env python

import rospy
import tf2_ros
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from std_msgs.msg import Header, Bool
import math
import tf
from geometry_msgs.msg import Pose, Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def GetTF(target_frame: str, source_frame: str):
    while True:
        try:
            trans = tf_buffer.lookup_transform(target_frame, source_frame, time=rospy.Time(0))

            return trans
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.sleep(0.01)
            continue



if __name__ == '__main__':
    
    rospy.init_node('object_tf_listener')
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    object_pose = rospy.Publisher('/object_pose', Pose,queue_size=10)

    while not rospy.is_shutdown():
        rate = rospy.Rate(10.0)

        trans = GetTF('bottle1','base_link')
        
        pose = Pose()
        pose.position = Point(trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z)
        pose.orientation = trans.transform.rotation
        object_pose.publish(pose)
        rate.sleep()