#! /usr/bin/env python

PACKAGE='python_pick_and_place_test'
import roslib
roslib.load_manifest(PACKAGE)
import rospy
import sys

from moveit_commander import MoveGroupCommander
from math import pi

class PickAndPlace():
    # Must have __init__(self) function for a class, similar to a C++ class constructor.
    def __init__(self):
        group = MoveGroupCommander("arm")
        
        #group.set_orientation_tolerance([0.3,0.3,0,3])
        
        group.set_goal_tolerance(0.001)
        
        print "Planning frame: " ,group.get_planning_frame()
        print "Pose reference frame: ",group.get_pose_reference_frame()
        
        
        print "RPy target: 0,0,0"
        group.set_rpy_target([0,0,0],"katana_motor5_wrist_roll_link")
        group.go()
        print "Current rpy: " , group.get_current_rpy("katana_motor5_wrist_roll_link")
        
        raw_input()
        
        
        print "RPy target: pi,0,0"
        group.set_rpy_target([pi,0,0],"katana_motor5_wrist_roll_link")
        group.go()
        print "Current rpy: " , group.get_current_rpy("katana_motor5_wrist_roll_link")
        
        raw_input()
        
        print "RPy target: 0,0,-pi"
        group.set_rpy_target([0,0,-pi],"katana_motor5_wrist_roll_link")
        group.go()
        print "Current rpy: " , group.get_current_rpy("katana_motor5_wrist_roll_link")
        
        raw_input()
        
        print "RPy target: 0,0,pi"
        group.set_rpy_target([0,0,pi],"katana_motor5_wrist_roll_link")
        group.go()
        print "Current rpy: " , group.get_current_rpy("katana_motor5_wrist_roll_link")
        
        raw_input()
        
        print "RPy target: 0,pi,0"
        group.set_rpy_target([0,pi,0],"katana_motor5_wrist_roll_link")
        group.go()
        print "Current rpy: " , group.get_current_rpy("katana_motor5_wrist_roll_link")

        
        print "RPy target: 0,0,0"
        group.set_rpy_target([0,0,0],"katana_motor5_wrist_roll_link")
        group.go()
        print "Current rpy: " , group.get_current_rpy("katana_motor5_wrist_roll_link")
        
        raw_input()
        
        #group.set_position_target([0.35,0.0,0.85])
        #group.go()
        
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('pick_and_place_python')
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        ne = PickAndPlace()
    except rospy.ROSInterruptException: pass