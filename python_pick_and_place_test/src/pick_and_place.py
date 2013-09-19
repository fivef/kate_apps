#! /usr/bin/env python

PACKAGE='python_pick_and_place_test'
import roslib
roslib.load_manifest(PACKAGE)
from geometry_msgs.msg import PoseStamped
import rospy
import sys

from moveit_commander import MoveGroupCommander


from math import pi
from math import atan2
import tf
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion


class PickAndPlace():
    # Must have __init__(self) function for a class, similar to a C++ class constructor.
    def __init__(self):
        group = MoveGroupCommander("arm")
        
        #group.set_orientation_tolerance([0.3,0.3,0,3])
        
        p = PoseStamped()
        p.header.frame_id = "/katana_base_link"
        p.pose.position.x = 0.4287
        #p.pose.position.y = -0.0847
        p.pose.position.y = 0.0
        p.pose.position.z = 0.4492
        
        
        q = quaternion_from_euler(2, 0, 0)
        
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]
        


        
        print "Planning frame: " ,group.get_planning_frame()
        print "Pose reference frame: ",group.get_pose_reference_frame()
        
        #group.set_pose_reference_frame("katana_base_link")

        print "RPy target: 0,0,0"
        #group.set_rpy_target([0, 0, 0],"katana_gripper_tool_frame")
        #group.set_position_target([0.16,0,0.40], "katana_gripper_tool_frame")
        
        group.set_pose_target(p, "katana_gripper_tool_frame")
        
        group.go()
        print "Current rpy: " , group.get_current_rpy("katana_gripper_tool_frame")
        
        
    def replace_yaw_angle_with_reachable_value(self, pose_stamped):
      #print "Original pose" , pose_stamped
      euler_orientation = list(euler_from_quaternion([pose_stamped.pose.orientation.x,pose_stamped.pose.orientation.y,pose_stamped.pose.orientation.z,pose_stamped.pose.orientation.w]))
      euler_orientation[2] = atan2(pose_stamped.pose.position.y, pose_stamped.pose.position.x)
      
      quaternion = quaternion_from_euler(euler_orientation[0], euler_orientation[1], euler_orientation[2])
      pose_stamped.pose.orientation.x = quaternion[0]
      pose_stamped.pose.orientation.y = quaternion[1]
      pose_stamped.pose.orientation.z = quaternion[2]
      pose_stamped.pose.orientation.w = quaternion[3]
      
      #print "Adjusted yaw: " , pose_stamped
        
        
        
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('pick_and_place_python')
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        ne = PickAndPlace()
    except rospy.ROSInterruptException: pass