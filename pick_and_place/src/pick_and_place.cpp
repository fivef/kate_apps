#include <ros/ros.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <actionlib/client/simple_action_client.h>

#include <std_srvs/Empty.h>
#include "geometry_msgs/Point.h"
#include <boost/spirit/include/classic.hpp>
#include "visualization_msgs/Marker.h"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <math.h>
#include <sstream>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>

#include <ros/callback_queue.h>

//#include <object_manipulation_msgs/>
#include <object_manipulation_msgs/FindClusterBoundingBox2.h>
#include <tabletop_object_detector/TabletopDetection.h>
#include <tabletop_collision_map_processing/TabletopCollisionMapProcessing.h>

#include <manipulation_msgs/GraspPlanning.h>
#include <manipulation_msgs/GraspPlanningRequest.h>
#include <manipulation_msgs/GraspPlanningAction.h>
#include <manipulation_msgs/GraspPlanningGoal.h>

#include <moveit/pick_place/manipulation_stage.h>

//for PTU dynamic reconfigure
#include <dynamic_reconfigure/DoubleParameter.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>

#include <interactive_markers/interactive_marker_server.h>
#include <interactive_markers/interactive_marker_client.h>
#include <interactive_markers/menu_handler.h>

using namespace visualization_msgs;

// MoveIt!
#include <moveit/pick_place/pick_place.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/move_group_interface/move_group.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <moveit_msgs/PlanningSceneComponents.h>
#include <moveit_msgs/PickupAction.h>
#include <moveit_msgs/PickupGoal.h>
#include <moveit_msgs/ExecuteKnownTrajectory.h>

#include <shape_tools/solid_primitive_dims.h>

// PCL specific includes
#include <pcl-1.6/pcl/common/eigen.h>
#include <pcl-1.6/pcl/ros/conversions.h>
#include <pcl-1.6/pcl/point_cloud.h>
#include <pcl-1.6/pcl/point_types.h>
#include <pcl-1.6/pcl/features/normal_3d.h>
#include <pcl-1.6/pcl/pcl_base.h>
#include <pcl-1.6/pcl/kdtree/kdtree_flann.h>

#include <pcl_ros/point_cloud.h>

#include <tf2/LinearMath/btVector3.h>
#include <tf2/LinearMath/btQuaternion.h>

//set service and action names
const std::string OBJECT_DETECTION_SERVICE_NAME = "/object_detection";
const std::string COLLISION_PROCESSING_SERVICE_NAME =
		"/tabletop_collision_map_processing/tabletop_collision_map_processing";
const std::string PICKUP_ACTION_NAME = "/pickup";
const std::string GET_PLANNING_SCENE_SERVICE_NAME = "/get_planning_scene";
const std::string PLAN_POINT_CLUSTER_GRASP_SERVICE_NAME =
		"/plan_point_cluster_grasp";
const std::string EVALUATE_POINT_CLUSTER_GRASP_SERVICE_NAME =
		"/evaluate_point_cluster_grasps";

const std::string EXECUTE_KINEMATIC_PATH_SERVICE_NAME = "/execute_kinematic_path";


//! General base class for all exceptions originating in the collision map interface
class CollisionMapException: public std::runtime_error {
public:
	CollisionMapException(const std::string error) :
			std::runtime_error("collision map: " + error) {
	}
	;
};

class Pick_and_place_app {

private:
	static const int DEBUG = true;

	static const float place_position_tolerance_in_meter = 0.03;
	static const float place_planner_step_size_in_meter = 0.005;

	//the distance in y direction from the position where the object was picked up to the place position
	static const float place_offset = 0.20;

	static const size_t NUM_JOINTS = 5;

	static const double gripper_open = -0.5;
	static const double gripper_closed = 0.2;

	double STANDOFF;

	std::string ARM_BASE_LINK;

	std::string BASE_LINK;

	std::string GRIPPER_FRAME;

	std::string FINGER_JOINT;

	std::string ARM_NAME;

	static const int MOVE_TO = 0;
	static const int PICKUP_PLAN_ONLY = 1;
	static const int PICKUP_MANUAL = 2;
	std::string TEST_MODE;

	geometry_msgs::PointStamped desired_pickup_point;

	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input_point_cloud;

	int running;

	ros::Publisher vis_marker_publisher;

	tf::TransformListener *tf_listener;

	geometry_msgs::PoseStamped normalPoseRobotFrame;

	pcl::PointXYZ normal;

	geometry_msgs::PoseStamped currentMarkerPose;

	std::vector<manipulation_msgs::Grasp> pickup_grasps;

	moveit_msgs::CollisionObject object_to_manipulate_object;
	std::string object_to_manipulate;
	int object_to_manipulate_index;

	geometry_msgs::Point object_to_manipulate_position;

	move_group_interface::MoveGroup *group;

	move_group_interface::MoveGroup *gripper;

	planning_scene_monitor::PlanningSceneMonitor *planningSceneMonitor;

	ros::Publisher pub_collision_object;

	std::vector<geometry_msgs::PointStamped> object_positions;

	std::vector<manipulation_msgs::GraspableObject> graspable_objects;

	std::vector<std::string> collision_object_names;

	//index of the object to pick up
	int object_to_pick_ind;

	//results of pickup needed for place
	//object_manipulation_msgs::PickupResult pickup_result;

	//constraints for moving arm out of the way
	//std::vector<arm_navigation_msgs::JointConstraint> *joint_constraints;

	// processing_call.response contains the graspable objects
	//tabletop_collision_map_processing::TabletopCollisionMapProcessing processing_call;

	//Is gazebo simulation? Is set from parameter server param: sim
	int sim;

	// processing_call.response contains the graspable objects
	tabletop_collision_map_processing::TabletopCollisionMapProcessing processing_call;

	ros::ServiceClient cluster_bounding_box2_3d_client_;

	//service and action clients
	ros::ServiceClient object_detection_srv;
	ros::ServiceClient collision_processing_srv;
	ros::ServiceClient collider_reset_srv;
	ros::ServiceClient get_planning_scene_srv;
	ros::ServiceClient evaluate_point_cluster_grasp_srv_client;
	ros::ServiceClient execute_kinematic_path_srv;

	ros::Publisher attached_object_publisher;

	actionlib::SimpleActionClient<manipulation_msgs::GraspPlanningAction> plan_point_cluster_grasp_action_client;

	actionlib::SimpleActionClient<moveit_msgs::PickupAction> pickup_action_client;

	ros::NodeHandle nh;

	boost::shared_ptr<interactive_markers::InteractiveMarkerServer> server;
	interactive_markers::MenuHandler menu_handler;


public:

	int object_in_gripper;
	//only calculate normal once after click
	int clicked;

	Pick_and_place_app(ros::NodeHandle *_nh) :
			plan_point_cluster_grasp_action_client(
					PLAN_POINT_CLUSTER_GRASP_SERVICE_NAME, true),
			pickup_action_client(PICKUP_ACTION_NAME, true)

	{

		nh = *_nh;

		//get parameters from parameter server
		nh.param<int>("sim", sim, 0);

		//the distance between the surface of the object to grasp and the GRIPPER_FRAME origin
		nh.param<double>("OBJECT_GRIPPER_STANDOFF", STANDOFF, -0.02);

		nh.param<std::string>("ARM_BASE_LINK", ARM_BASE_LINK, "jaco_base_link");

		nh.param<std::string>("BASE_LINK", BASE_LINK, "/base_link");

		nh.param<std::string>("GRIPPER_FRAME", GRIPPER_FRAME,
				"jaco_gripper_tool_frame");

		//the finger joint name without the number (_1) at the end
		nh.param<std::string>("FINGER_JOINT", FINGER_JOINT,
				"jaco_finger_joint");
		nh.param<std::string>("ARM_NAME", ARM_NAME, "arm");

		clicked = 0;

		object_in_gripper = 0;

		pcl_input_point_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
				new pcl::PointCloud<pcl::PointXYZ>());

		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

		// create TF listener
		tf_listener = new tf::TransformListener();

		pub_collision_object = nh.advertise<moveit_msgs::CollisionObject>(
				"collision_object", 10);

		ros::WallDuration(1.0).sleep();

		group = new move_group_interface::MoveGroup(ARM_NAME);

		gripper = new move_group_interface::MoveGroup("gripper");

		planningSceneMonitor = new planning_scene_monitor::PlanningSceneMonitor("robot_description");

		//group->setPoseReferenceFrame(BASE_LINK);

		group->setPlanningTime(20.0);



		//TODO check / comment
		//group->setGoalTolerance(10);
		//group->setGoalOrientationTolerance(0.005);
		//group->setPlannerId("RRTConnectkConfigDefault");

		ROS_INFO_STREAM("Planning time: " << group->getPlanningTime());
		group->setPlannerId("RRTConnectkConfigDefault");
		                            //RRTConnectkConfigDefault
		group->setPoseReferenceFrame(BASE_LINK);

		group->allowReplanning(true);

		group->setEndEffector("gripper");

		group->setEndEffectorLink(GRIPPER_FRAME);

		group->setWorkspace(-0.5, -0.6, -0.3, 0.6, 0.6, 1.5);





		//wait for get planning scene server

		while (!ros::service::waitForService(GET_PLANNING_SCENE_SERVICE_NAME,
				ros::Duration(2.0)) && nh.ok()) {
			ROS_INFO("Waiting for get planning scene service to come up");
		}
		if (!nh.ok())
			exit(0);
		get_planning_scene_srv =
				nh.serviceClient<moveit_msgs::GetPlanningScene>(
						GET_PLANNING_SCENE_SERVICE_NAME, true);

		//wait for detection client
		while (!ros::service::waitForService(OBJECT_DETECTION_SERVICE_NAME,
				ros::Duration(2.0)) && nh.ok()) {
			ROS_INFO("Waiting for object detection service to come up");
		}
		if (!nh.ok())
			exit(0);
		object_detection_srv = nh.serviceClient<
				tabletop_object_detector::TabletopDetection>(
				OBJECT_DETECTION_SERVICE_NAME, true);

		//wait for collision map processing client
		while (!ros::service::waitForService(COLLISION_PROCESSING_SERVICE_NAME,
				ros::Duration(2.0)) && nh.ok()) {
			ROS_INFO("Waiting for collision processing service to come up");
		}
		if (!nh.ok())
			exit(0);

		collision_processing_srv =
				nh.serviceClient<
						tabletop_collision_map_processing::TabletopCollisionMapProcessing>(
						COLLISION_PROCESSING_SERVICE_NAME, true);

		evaluate_point_cluster_grasp_srv_client = nh.serviceClient<
				manipulation_msgs::GraspPlanning>(
				EVALUATE_POINT_CLUSTER_GRASP_SERVICE_NAME, true);

		execute_kinematic_path_srv = nh.serviceClient<moveit_msgs::ExecuteKnownTrajectory>(EXECUTE_KINEMATIC_PATH_SERVICE_NAME);

		vis_marker_publisher = nh.advertise<visualization_msgs::Marker>(
				"pick_and_place_markers", 128);

		attached_object_publisher = nh.advertise<moveit_msgs::AttachedCollisionObject>("attached_collision_object", 1);

		//ineractive marker server
		server.reset(
				new interactive_markers::InteractiveMarkerServer(
						"basic_controls", "", true));

		make6DofMarker();

		server->applyChanges();

		//Sets the kinects tilt angle

		set_kinect_ptu("kurtana_pitch_joint", 0.75);

		ROS_INFO("Kinect lined up.");

		move_arm_out_of_the_way();

	}

	~Pick_and_place_app() {

		//Sets the kinects tilt angle
		set_kinect_ptu("kurtana_pitch_joint", 0.0);

	}

	void set_pickup_point(geometry_msgs::PointStamped point_) {

		desired_pickup_point = point_;

		ROS_INFO_STREAM(
				"Pickup Point set to: " << desired_pickup_point.point.x << " " << desired_pickup_point.point.y << " " << desired_pickup_point.point.z);

	}

	void set_kinect_ptu(std::string joint_name, double value) {

		dynamic_reconfigure::ReconfigureRequest srv_req;
		dynamic_reconfigure::ReconfigureResponse srv_resp;
		dynamic_reconfigure::DoubleParameter double_param;
		dynamic_reconfigure::Config conf;

		double_param.name = joint_name;
		double_param.value = value;
		conf.doubles.push_back(double_param);

		srv_req.config = conf;

		ros::service::call("/joint_commander/set_parameters", srv_req,
				srv_resp);
	}

	geometry_msgs::PointStamped get_pickup_point() {

		return desired_pickup_point;
	}

	void pick_and_place() {

		//Sets the kinects tilt angle
		set_kinect_ptu("kurtana_pitch_joint", 0.85);

		if (move_arm_out_of_the_way()) {

			if (move_arm_out_of_the_way()) {
				return;
			}

		}

		if (pickup()) {

			if (pickup()) {
				//don't place if pickup failed
				return;
			}
		}

		if (place()) {

			if (place()) {

				return;
			}
			place();
		}

		if (move_arm_out_of_the_way()) {

			if (move_arm_out_of_the_way()) {
				return;
			}

		}

		return;

	}

	int pickup_nearest_segmented_object() {

		ROS_INFO_STREAM("Picking up nearest segmented object");

		if(detect_objects_on_table() != 0){

			ROS_INFO("Object Detection Failed");

			return -1;

		}

		find_nearest_object();

		generate_grasps_for_nearest_cluster();

		ROS_INFO_STREAM(
				"Picking up Object: " << object_to_manipulate << " with " << pickup_grasps.size() << " grasps to try");

		manipulation_msgs::Grasp normal_grasp = generateGraspFromNormal();


		//pickup_grasps.push_back(normal_grasp);
		pickup_grasps.at(0) = normal_grasp;

		setMarkerToPoseStamped(normal_grasp.grasp_pose);

		draw_pickup_grasps_to_try();

		pickup_plan_only();

		object_in_gripper = 1;

		return 0;
	}

	int pickup_with_current_marker_pose() {

		generate_grasps_based_on_current_marker_pose();

		//create_dummy_collision_object(currentMarkerPose);

		pickup();

		return 0;
	}

	int move_to_current_marker_pose() {

		pickup_grasps.resize(1);
		pickup_grasps[0] = create_grasp_out_of_pose_stamped(currentMarkerPose);

		group->setPoseTarget(currentMarkerPose);

		group->move();




		return 0;
	}

	int pickup_with_normal() {

		ROS_INFO_STREAM("Picking up by surface normal");

		ROS_INFO("Calling the pickup action");

		pickup_grasps.resize(1);
		pickup_grasps[0] = generateGraspFromNormal();

		setMarkerPoseToFirstGrasp();

		create_dummy_collision_object(pickup_grasps[0].grasp_pose);

		pickup();

		return 0;
	}

	int pickup() {

		//group->setSupportSurfaceName("table");

		//draw_grasps_to_try();

		group->pick(object_to_manipulate, pickup_grasps);
		//pickup_plan_only();


		object_in_gripper = 1;

		//move_arm_out_of_the_way();

		return 0;
	}

	bool open_gripper(bool plan_only = false){

		gripper->setJointValueTarget("jaco_joint_6",gripper->getCurrentJointValues()[0]);

		gripper->setJointValueTarget(FINGER_JOINT + "_1", gripper_open);
		gripper->setJointValueTarget(FINGER_JOINT + "_2", gripper_open);
		gripper->setJointValueTarget(FINGER_JOINT + "_3", gripper_open);

		if(plan_only){

			moveit::planning_interface::MoveGroup::Plan plan;
			if(!gripper->plan(plan)) return false;

		}else{
			gripper->move();
		}


		return true;

	}

	bool close_gripper(bool plan_only = false){

		gripper->setJointValueTarget("jaco_joint_6",gripper->getCurrentJointValues()[0]);

		gripper->setJointValueTarget(FINGER_JOINT + "_1", gripper_closed);
		gripper->setJointValueTarget(FINGER_JOINT + "_2", gripper_closed);
		gripper->setJointValueTarget(FINGER_JOINT + "_3", gripper_closed);

		if(plan_only){

			moveit::planning_interface::MoveGroup::Plan plan;
			if(!gripper->plan(plan)) return false;

		}else{
			gripper->move();
		}

		return true;

	}

	bool pickup_manually(bool plan_only = false){

		geometry_msgs::PoseStamped graspPose = currentMarkerPose;
		geometry_msgs::PoseStamped preGraspPose;
		geometry_msgs::PoseStamped retreatPose;
		preGraspPose.header = graspPose.header;
		retreatPose.header = graspPose.header;

		//create_dummy_collision_object(graspPose);

		//Shift graspPose -0.1 m to get the preGraspPose
		tf::Transform position;
		tf::Transform standoff;
		tf::Transform rotation;

		tf::Quaternion quaternion;

		ROS_INFO_STREAM("pose before msg to tf " << graspPose.pose);

		tf::quaternionMsgToTF(graspPose.pose.orientation,quaternion);

		position.setIdentity();
		position.setOrigin(tf::Vector3(graspPose.pose.position.x, graspPose.pose.position.y, graspPose.pose.position.z));

		rotation.setIdentity();
		rotation.setRotation(quaternion);

		standoff.setIdentity();
		standoff.setOrigin(tf::Vector3(-0.1,0,0));

		geometry_msgs::Pose shifted_pose;

		tf::poseTFToMsg(position * rotation * standoff, shifted_pose);

		preGraspPose.pose = shifted_pose;


		ROS_INFO("//PHASE 1 // moving to pregrasp pose");

		group->setStartStateToCurrentState();

		group->setPoseTarget(preGraspPose, GRIPPER_FRAME);


		moveit::planning_interface::MoveGroup::Plan plan;

		if(plan_only){

			if(!group->plan(plan)){
				ROS_INFO("Pregrasp failed");
				return false;
			}
		}else{
			group->move();

		}

		//setMarkerToPose(pose);

		ROS_INFO("//PHASE 2 // open gripper");

		if(open_gripper(plan_only)){

		}else{
			ROS_INFO("Open gripper failed");
			return false;
		}

		ROS_INFO_STREAM("Collision matrix size " << planningSceneMonitor->getPlanningScene()->getAllowedCollisionMatrix().getSize());

		//planningSceneMonitor->getPlanningScene()->getAllowedCollisionMatrixNonConst().setEntry("dummy","jaco_finger_joint_1",true);

		planning_scene::PlanningScenePtr planning_scene = planningSceneMonitor->getPlanningScene();

		//collision_detection::AllowedCollisionMatrixPtr approach_grasp_acm(new collision_detection::AllowedCollisionMatrix(planning_scene->getAllowedCollisionMatrix()));

		//const robot_model::JointModelGroup *eef = planning_scene->getRobotModel()->getEndEffector("gripper");

		//approach_grasp_acm->setEntry("dummy", eef->getLinkModelNames(), true);
		//planningSceneMonitor->getPlanningScene()->getAllowedCollisionMatrixNonConst().setEntry("dummy",FINGER_JOINT + "_1",true);


		ROS_INFO("//PHASE 3 // moving to grasp pose");

		std::vector<geometry_msgs::Pose> waypoints;
		waypoints.push_back(preGraspPose.pose);
		waypoints.push_back(graspPose.pose);

		moveit_msgs::RobotTrajectory trajectory;

		moveit_msgs::ExecuteKnownTrajectory srv;

		double eef_step_size = 0.01;
		double jump_factor= 10000;
		bool avoid_collisions = false;

		/*
		 * Compute a Cartesian path that follows specified waypoints with a step size of at most eef_step meters
		 * between end effector configurations of consecutive points in the result trajectory. The reference frame
		 * for the waypoints is that specified by setPoseReferenceFrame(). No more than jump_threshold is allowed
		 * as change in distance in the configuration space of the robot (this is to prevent 'jumps' in IK solutions).
		 * Collisions are avoided if avoid_collisions is set to true. If collisions cannot be avoided, the function fails.
		 * Return a value that is between 0.0 and 1.0 indicating the fraction of the path achieved as described by the waypoints.
		 * Return -1.0 in case of error.
		 */
		double fraction_of_path_achieved = group->computeCartesianPath(waypoints, eef_step_size, jump_factor, srv.request.trajectory, avoid_collisions);

		if(fraction_of_path_achieved < 0.9){
			ROS_INFO_STREAM("Failed at approach Cartesian path: " << fraction_of_path_achieved << " < 0.9");
			return false;
		}

		if(!plan_only){
			srv.request.wait_for_execution = true;
			execute_kinematic_path_srv.call(srv);
		}

		//Remove the object to grasp to allow the gripper to close without collisions
		//it would be better to disable collisions completely here
		remove_collision_object_from_scene(object_to_manipulate_object);

		ROS_INFO("//PHASE 4 // close gripper and attach object");
		if(close_gripper(plan_only)){

		}else{
			ROS_INFO("Close gripper failed");
			return false;
		}

		attach_object_to_gripper(object_to_manipulate_object);

		ROS_INFO("//PHASE 5 // retreat");

		//reusing the objects from approach

		position.setIdentity();
		position.setOrigin(tf::Vector3(graspPose.pose.position.x, graspPose.pose.position.y, graspPose.pose.position.z));

		standoff.setIdentity();
		standoff.setOrigin(tf::Vector3(0,0,0.1));

		tf::poseTFToMsg(standoff * position, shifted_pose);

		retreatPose.pose = shifted_pose;

		//compute cartesian path for retreat reusing the objects from above
		waypoints.resize(0);
		waypoints.push_back(graspPose.pose);
		waypoints.push_back(retreatPose.pose);

		fraction_of_path_achieved = group->computeCartesianPath(waypoints, eef_step_size, jump_factor, srv.request.trajectory, avoid_collisions);

		if(fraction_of_path_achieved < 0.5){
			ROS_INFO_STREAM("Failed at retreat Cartesian path: " << fraction_of_path_achieved << " < 0.9");
			return false;
		}

		if(!plan_only){
			srv.request.wait_for_execution = true;
			execute_kinematic_path_srv.call(srv);
		}

		//Check if the whole grasping process was successful by comparing the retreatPose with the current pose
		if(!plan_only){
			if(compareTwoValuesWithThreshold(retreatPose.pose.position.z,group->getCurrentPose().pose.position.z)){
				if(compareTwoValuesWithThreshold(retreatPose.pose.position.y,group->getCurrentPose().pose.position.y)){
					if(compareTwoValuesWithThreshold(retreatPose.pose.position.x,group->getCurrentPose().pose.position.x)){
						return true;
					}

				}
			}

			return false;
		}

		return true;
	}


	bool compareTwoValuesWithThreshold(double value, double value_to_compare_to){

		double threshold = 0.06;

		if(value > (value_to_compare_to - threshold)){
			if(value < (value_to_compare_to + threshold)){
				return true;
			}
		}

		return false;
	}

	int attach_object_to_gripper(moveit_msgs::CollisionObject collision_object){

		while(attached_object_publisher.getNumSubscribers() < 1)
		{
		ros::WallDuration sleep_t(0.5);
		sleep_t.sleep();
		}

		moveit_msgs::AttachedCollisionObject attached_object;
		attached_object.link_name = GRIPPER_FRAME;

		attached_object.object = collision_object;

		attached_object.touch_links.push_back("jaco_finger_link_1");
		attached_object.touch_links.push_back("jaco_finger_link_2");
		attached_object.touch_links.push_back("jaco_finger_link_3");
		attached_object.touch_links.push_back("jaco_link_6");


		attached_object_publisher.publish(attached_object);

		return 1;
	}

	int pickup_plan_only(){

		ROS_INFO("Pickup plan only");

		//pickup_grasps[0].

		moveit_msgs::PickupGoal pickupGoal;

		pickupGoal.target_name = object_to_manipulate;
		pickupGoal.group_name = ARM_NAME;
		pickupGoal.end_effector = "gripper";
		pickupGoal.support_surface_name = "table";
		pickupGoal.allowed_planning_time = 20.0;
		pickupGoal.planner_id = "arm[RRTConnectkConfigDefault]";
		//pickupGoal.minimize_object_distance = true; Does this work?

		moveit_msgs::PlanningOptions planning_options;
		planning_options.plan_only = true;


		pickupGoal.planning_options = planning_options;


		//pickupGoal.header;
		//pickupGoal.

		pickupGoal.possible_grasps = pickup_grasps;

		//ROS_DEBUG_STREAM("Pickup Plan: Pickup Grasps size: " << pickup_grasps.size());

		//ROS_DEBUG_STREAM("Pickup Plan: Grasp Pose: " << pickup_grasps[0].grasp_pose);

		pickup_action_client.sendGoal(pickupGoal);

		//pickup_action_client.sendGoal(pickupGoal, pickupResult, pickupFeedback);

		//wait for the action to return
		bool finished_before_timeout =
				pickup_action_client.waitForResult(
						ros::Duration(30.0));

		if (finished_before_timeout) {
			ROS_INFO("Received Response!");
			actionlib::SimpleClientGoalState state =
					pickup_action_client.getState();
			ROS_INFO("Action finished: %s", state.toString().c_str());

			if(state == actionlib::SimpleClientGoalState::SUCCEEDED){
				return 1;
			}else return 0;


		} else
			ROS_INFO("Action did not finish before the time out.");

		//exit
		return 0;


	}


	/*
	void draw_selected_cluster() {

		std::vector<geometry_msgs::Point32> points =
				graspable_objects[object_to_manipulate_index].points;

		//draw cluster
		visualization_msgs::Marker marker;

		ROS_INFO_STREAM("Number of points in cluster: " << points.size());

		for (size_t i = 0; i < points.size(); i++) {
			marker.pose.position.x = points[i].x;
			marker.pose.position.y = points[i].y;
			marker.pose.position.z = points[i].z;

			//show markers in kinect frame
			marker.header.frame_id =
					graspable_objects[object_to_manipulate_index].header.frame_id;
			marker.id = i;
			marker.ns = "cluster";
			marker.header.stamp = ros::Time::now();
			marker.action = visualization_msgs::Marker::ADD;
			marker.lifetime = ros::Duration();
			marker.type = visualization_msgs::Marker::SPHERE;
			marker.scale.x = 0.002;
			marker.scale.y = 0.002;
			marker.scale.z = 0.002;
			marker.color.r = 0;
			marker.color.g = 0;
			marker.color.b = 1;
			marker.color.a = 1.0;
			vis_marker_publisher.publish(marker);

		}

	}*/

	int generate_grasps_for_nearest_cluster() {

		ROS_INFO("Generateing grasps for nearest cluster");
		//draw_selected_cluster();

		//Get grasps

		manipulation_msgs::GraspPlanningGoal graspPlanningGoal;

		ROS_INFO_STREAM("Generateing grasps for object " << object_to_manipulate_index << " of " << graspable_objects.size());

		//graspPlanningGoal.target.cluster =
				//graspable_objects[object_to_manipulate_index];

		if(graspable_objects.size() == 0){

			ROS_ERROR("No objects available");

			return 0;
		}

		graspPlanningGoal.target = graspable_objects[object_to_manipulate_index];

		graspPlanningGoal.arm_name = ARM_NAME;

		graspPlanningGoal.collision_support_surface_name = "table";

		graspPlanningGoal.collision_object_name = object_to_manipulate;

		plan_point_cluster_grasp_action_client.sendGoal(graspPlanningGoal);

		//wait for the action to return
		bool finished_before_timeout =
				plan_point_cluster_grasp_action_client.waitForResult(
						ros::Duration(30.0));

		if (finished_before_timeout) {
			ROS_INFO("Received Response!");
			actionlib::SimpleClientGoalState state =
					plan_point_cluster_grasp_action_client.getState();
			ROS_INFO("Action finished: %s", state.toString().c_str());

			pickup_grasps.resize(0);
			pickup_grasps =
					plan_point_cluster_grasp_action_client.getResult()->grasps;

			return 1;

			/*
			std::vector<manipulation_msgs::Grasp> generated_grasps;

			for(int i = 0; i < plan_point_cluster_grasp_action_client.getResult()->grasps.size(); i++){

				generated_grasps = generate_grasps_for_pose_stamped(plan_point_cluster_grasp_action_client.getResult()->grasps[i].grasp_pose);
				pickup_grasps.insert(pickup_grasps.end(),generated_grasps.begin(), generated_grasps.end());
			}

			ROS_INFO_STREAM("Total grasps to try: " << pickup_grasps.size());

			*/

		} else
			ROS_INFO("Action did not finish before the time out.");

		//exit
		return 0;
	}

	int detect_objects_on_table() {

		remove_all_scene_objects();

		// ----- call the tabletop detection
		ROS_INFO("Calling tabletop detector");
		tabletop_object_detector::TabletopDetection detection_call;
		//we want recognized database objects returned
		//set this to false if you are using the pipeline without the database
		detection_call.request.return_models = false;

		//we want the individual object point clouds returned as well
		detection_call.request.return_clusters = true;

		if (!object_detection_srv.call(detection_call)) {
			ROS_ERROR("Tabletop detection service failed");
			return -1;
		}
		if (detection_call.response.detection.result
				!= detection_call.response.detection.SUCCESS) {
			ROS_ERROR(
					"Tabletop detection returned error code %d", detection_call.response.detection.result);
			return -1;
		}
		if (detection_call.response.detection.clusters.empty()
				&& detection_call.response.detection.models.empty()) {
			ROS_ERROR(
					"The tabletop detector detected the table, but found no objects");
			return -1;
		}



		//Remove the table because there are convex hull problems if adding the table to envirnonment
		//detection_call.response.detection.table.convex_hull = shape_msgs::Mesh();

		group->setSupportSurfaceName("table");

		tabletop_collision_map_processing::TabletopCollisionMapProcessing process_call;

		process_call.request.detection_result =
				detection_call.response.detection;
		process_call.request.reset_collision_models = false;
		process_call.request.reset_attached_models = false;


		if (!collision_processing_srv.call(process_call)) {
			ROS_ERROR("Tabletop Collision Map Processing failed");
			return -1;
		}

		ROS_INFO_STREAM(
				"Found objects count: " << process_call.response.collision_object_names.size());

		collision_object_names = process_call.response.collision_object_names;



		//save clusters for later

		graspable_objects = process_call.response.graspable_objects;
		ROS_INFO_STREAM(
				"Number of clusters found: " << graspable_objects.size());

		if (process_call.response.collision_object_names.empty()) {
			ROS_ERROR("Tabletop Collision Map Processing error");
			return -1;
		}

		return 0;

	}

	void remove_collision_object_from_scene(moveit_msgs::CollisionObject collision_object){

		collision_object.operation = moveit_msgs::CollisionObject::REMOVE;

		pub_collision_object.publish(collision_object);
	}

	/*
	 * creates a dummy collision object for the pick() function
	 * Takes the pickup_grasp to set the objects pose.
	 */
	void create_dummy_collision_object(geometry_msgs::PoseStamped pose) {

		tf::Transform position;
		tf::Transform standoff;
		tf::Transform rotation;

		tf::Quaternion quaternion;
		tf::quaternionMsgToTF(pose.pose.orientation,quaternion);

		position.setIdentity();
		position.setOrigin(tf::Vector3(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z));

		rotation.setIdentity();
		rotation.setRotation(quaternion);

		standoff.setIdentity();
		standoff.setOrigin(tf::Vector3(0.05,0,0));

		geometry_msgs::Pose shifted_pose;

		tf::poseTFToMsg(position * rotation * standoff, shifted_pose);

		pose.pose = shifted_pose;

		moveit_msgs::CollisionObject collision_object;

		collision_object.header.stamp = ros::Time::now();
		collision_object.header.frame_id = BASE_LINK;

		//add object

		collision_object.id = "dummy";

		collision_object.operation = moveit_msgs::CollisionObject::REMOVE;

		pub_collision_object.publish(collision_object);

		collision_object.operation = moveit_msgs::CollisionObject::ADD;

		collision_object.primitives.resize(1);
		collision_object.primitives[0].type = shape_msgs::SolidPrimitive::BOX;
		collision_object.primitives[0].dimensions.resize(
				shape_tools::SolidPrimitiveDimCount<
						shape_msgs::SolidPrimitive::BOX>::value);
		collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_X] =
				0.05;
		collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_Y] =
				0.05;
		collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_Z] =
				0.1;
		collision_object.primitive_poses.resize(1);
		collision_object.primitive_poses[0] = pose.pose;

		pub_collision_object.publish(collision_object);

		//set the current object to manipulate to the generated dummy object
		object_to_manipulate = "dummy";

	}

	void do_reachability_test(int type){

		moveit::planning_interface::MoveGroup::Plan plan;


		std::vector<manipulation_msgs::Grasp> test_grasps = generate_test_grasps();

		draw_grasps(test_grasps);

		int successful_grasps_counter = 0;

		ROS_INFO_STREAM("Test grasps type " << type);

		for(size_t i = 0 ; i < test_grasps.size(); i++){

			pickup_grasps.resize(1);
			pickup_grasps[0] = test_grasps[i];

			ROS_INFO_STREAM("Test grasp no " << i);

			ROS_INFO_STREAM("Grasp Pose: " << pickup_grasps[0].grasp_pose);

			switch(type){

			case 0:
				//test moving to pose

				group->setPoseTarget(pickup_grasps[0].grasp_pose);



				if(group->plan(plan)){

					successful_grasps_counter++;

					setSelectableMarkerToGreen(i);

				}else{
					setSelectableMarkerToRed(i);
				}

				break;

			case 1:

				pickup_grasps.resize(1);

				//test pickup_plan_only
				if(pickup_plan_only()){

					successful_grasps_counter++;

					setSelectableMarkerToGreen(i);

				}else{
					setSelectableMarkerToRed(i);
				}
				break;
			case 2:

				currentMarkerPose = pickup_grasps[0].grasp_pose;
				//pickup with manual pick function
				if(pickup_manually()){

					successful_grasps_counter++;

					setSelectableMarkerToGreen(i);

				}else{
					setSelectableMarkerToRed(i);
				}


				break;
			}


		}

		ROS_INFO_STREAM(successful_grasps_counter << " grasps out of " << test_grasps.size() << " were successful.");


	}

	/*
	 * generates a series of grasps for reachability testing
	 */
	std::vector<manipulation_msgs::Grasp> generate_test_grasps(){

		//startPoint and endPoint define a box.

		/*
		geometry_msgs::Point startPoint;
		startPoint.x = 0;
		startPoint.y = -1;
		startPoint.z = 0;
		geometry_msgs::Point endPoint;
		endPoint.x = 1;
		endPoint.y = 1;
		endPoint.z = 1.5;
		*/

		/*
		// good settings 225 grasps used for tests
		geometry_msgs::Point startPoint;
		startPoint.x = 0;
		startPoint.y = -0.9;
		startPoint.z = 0.4;
		geometry_msgs::Point endPoint;
		endPoint.x = 0.8;
		endPoint.y = 0.9;
		endPoint.z = 1.2;
		*/

		geometry_msgs::Point startPoint;
		startPoint.x = 0.4;
		startPoint.y = -0.6;
		startPoint.z = 0.5;
		geometry_msgs::Point endPoint;
		endPoint.x = 0.9;
		endPoint.y = 0.7;
		endPoint.z = 1.4;


		/*fewer grasps
		geometry_msgs::Point startPoint;
		startPoint.x = 0.4;
		startPoint.y = -0.7;
		startPoint.z = 0.4;
		geometry_msgs::Point endPoint;
		endPoint.x = 0.7;
		endPoint.y = 0.9;
		endPoint.z = 1.2;
		*/

		/* 16 grasps
		geometry_msgs::Point startPoint;
		startPoint.x = 0.7;
		startPoint.y = -0.2;
		startPoint.z = 0.8;
		geometry_msgs::Point endPoint;
		endPoint.x = 0.8;
		endPoint.y = 0.2;
		endPoint.z = 1.4;
		*/

		double step_size = 0.2;

		std::vector<manipulation_msgs::Grasp> generated_grasps;

		geometry_msgs::PoseStamped pose;


		tf::Transform pose_tf;

		pose_tf.setOrigin(tf::Vector3(0,0,0));
		pose_tf.setRotation(tf::createIdentityQuaternion());

		tf::Transform transform;

		transform.setRotation(tf::createIdentityQuaternion());


		for(double x = startPoint.x; x <= endPoint.x ; x += step_size ){

			for(double y = startPoint.y; y <= endPoint.y ; y += step_size ){

				for(double z = startPoint.z; z <= endPoint.z ; z += step_size ){

					transform.setOrigin(tf::Vector3(x,y,z));

					tf::poseTFToMsg(transform * pose_tf,pose.pose);

					pose.header.frame_id = BASE_LINK;

					ROS_INFO_STREAM("Grasp Pose: " << pose);

					generated_grasps.push_back(create_grasp_out_of_pose_stamped(pose));

				}
			}


		}

		ROS_INFO_STREAM(generated_grasps.size() << " test grasps generated.");

		return generated_grasps;
	}

	std::vector<manipulation_msgs::Grasp> generate_grasps_for_pose_stamped(geometry_msgs::PoseStamped pose_stamped){

		std::vector<manipulation_msgs::Grasp> generated_grasps;


		  static const double ANGLE_INC = M_PI / 20;

		  static const double ANGLE_MAX = M_PI / 18;


		  tf::Transform transform;

		  transform.setOrigin(tf::Vector3(pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z));

		  tf::Transform pose;

		  geometry_msgs::PoseStamped new_pose;

		  new_pose.header = pose_stamped.header;

		  tf::poseMsgToTF(pose_stamped.pose, pose);

		  pose.setOrigin(tf::Vector3(0, 0, 0));


			for (double yaw = -ANGLE_MAX; yaw <= ANGLE_MAX; yaw += ANGLE_INC)
			{



			  for (double pitch = -ANGLE_MAX; pitch <= ANGLE_MAX; pitch += ANGLE_INC)
			  {

				  for (double roll = -ANGLE_MAX; roll <= ANGLE_MAX; roll += ANGLE_INC)
				  {
					  transform.setRotation(tf::createQuaternionFromRPY(roll, pitch, yaw));

					  tf::poseTFToMsg(transform * pose,new_pose.pose);

					  generated_grasps.push_back(create_grasp_out_of_pose_stamped(new_pose));
				  }
			  }



			}


			ROS_INFO_STREAM(generated_grasps.size() << " grasp have been generated based on the pose");




		return generated_grasps;
	}


	void generate_grasps_based_on_current_marker_pose(){

		  pickup_grasps = generate_grasps_for_pose_stamped(currentMarkerPose);

		  ROS_INFO_STREAM(pickup_grasps.size() << " grasp have been generated based on the current marker pose");

	}



	/*
	 * Generate a grasp out of a pose
	 */

	manipulation_msgs::Grasp create_grasp_out_of_pose_stamped(
			geometry_msgs::PoseStamped pose) {



		manipulation_msgs::Grasp g;
		g.grasp_pose = pose;

		g.approach.direction.vector.x = 1.0;
		g.approach.direction.header.frame_id = GRIPPER_FRAME;
		g.approach.min_distance = 0.03;
		g.approach.desired_distance = 0.15;

		g.retreat.direction.header.frame_id = BASE_LINK;
		g.retreat.direction.vector.z = 1.0;
		g.retreat.min_distance = 0.03;
		g.retreat.desired_distance = 0.15;

		g.pre_grasp_posture.header.frame_id = BASE_LINK;
		g.pre_grasp_posture.header.stamp = ros::Time::now();
		g.pre_grasp_posture.name.resize(3);
		g.pre_grasp_posture.name[0] = FINGER_JOINT + "_1";
		g.pre_grasp_posture.position.resize(3);
		g.pre_grasp_posture.position[0] = gripper_open;

		//TODO create params for this
		g.pre_grasp_posture.name[1] = FINGER_JOINT + "_2";
		g.pre_grasp_posture.position[1] = gripper_open;

		g.pre_grasp_posture.name[2] = FINGER_JOINT + "_3";
		g.pre_grasp_posture.position[2] = gripper_open;

		g.grasp_posture.header.frame_id = BASE_LINK;
		g.grasp_posture.header.stamp = ros::Time::now();
		g.grasp_posture.name.resize(3);
		g.grasp_posture.name[0] = FINGER_JOINT + "_1";
		g.grasp_posture.position.resize(3);
		g.grasp_posture.position[0] = gripper_closed;

		g.grasp_posture.name[1] = FINGER_JOINT + "_2";
		g.grasp_posture.position[1] = gripper_closed;

		g.grasp_posture.name[2] = FINGER_JOINT + "_3";
		g.grasp_posture.position[2] = gripper_closed;




		//object allowd to be touche while approaching

		g.allowed_touch_objects.resize(1);
		g.allowed_touch_objects[0] = "all";

		g.grasp_quality = 1;

		//ROS_DEBUG_STREAM("Grasp frame id: " << g.grasp_pose.header.frame_id);

		//ROS_DEBUG_STREAM("Grasp Pose" << g.grasp_pose.pose);


		return g;

	}

	void draw_grasps(std::vector<manipulation_msgs::Grasp> grasps){

		ROS_INFO("Draw grasps");

		for (size_t i = 0; i < grasps.size(); i++) {


			makeSelectableMarker(i, grasps[i].grasp_pose);

		}


	}

	void draw_pickup_grasps_to_try() {

		ROS_INFO("Draw pickup grasps to try");

		draw_grasps(pickup_grasps);


	}

	/*
	 * Generates a Grasp out of the normalPoseRobotFrame pose.
	 */
	manipulation_msgs::Grasp generateGraspFromNormal() {

		tf::Vector3 position;
		position.setX(normalPoseRobotFrame.pose.position.x);
		position.setY(normalPoseRobotFrame.pose.position.y);
		position.setZ(normalPoseRobotFrame.pose.position.z);

		tf::Transform standoff_trans;
		standoff_trans.setOrigin(
				tf::Vector3(position.getX(), position.getY(), 0).normalize()
						* STANDOFF);
		standoff_trans.setRotation(tf::createIdentityQuaternion());

		//tf::poseStampedMsgToTF(normalPoseRobotFrame, position);

		position = standoff_trans * position;

		geometry_msgs::PoseStamped p;
		p.header.frame_id = BASE_LINK;
		p.pose.position.x = position.getX();
		p.pose.position.y = position.getY();
		p.pose.position.z = position.getZ();

		p.pose.orientation = normalPoseRobotFrame.pose.orientation;

		make_pose_reachable_by_5DOF_katana(p);

		return create_grasp_out_of_pose_stamped(p);
	}

	void make_pose_reachable_by_5DOF_katana(geometry_msgs::PoseStamped &pose) {

		//make the normal pose graspable by the Katana 5DOF gripper (Yaw missing)

		//input geometry_msgs::PoseStamped pose;
		//output geometry_msgs::PoseStamped fixed ppose;

		//Convert quaternion to RPY.

		tf::Quaternion q;

		double roll, pitch, yaw;
		tf::quaternionMsgToTF(pose.pose.orientation, q);

		/*
		 //TODO: check if quaternion is correct
		 if(q.length() < 0.99 || q.length() > 1.01){
		 q.setX(0.0);
		 q.setY(0.0);
		 q.setZ(0.0);
		 q.setW(1.0);
		 }
		 */

		tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

		//determine yaw which is compatible with the Katana 300 180 kinematics.
		yaw = atan2(pose.pose.position.y, pose.pose.position.x);

		tf::Quaternion quat = tf::createQuaternionFromRPY(0, pitch, yaw);

		pose.pose.orientation.x = quat.getX();
		pose.pose.orientation.y = quat.getY();
		pose.pose.orientation.z = quat.getZ();
		pose.pose.orientation.w = quat.getW();

	}

	bool pickup_callback(std_srvs::Empty::Request &request,
			std_srvs::Empty::Response &response) {
		if (pickup()) {
			return false;
		}
		return true;
	}

	bool place_callback(std_srvs::Empty::Request &request,
			std_srvs::Empty::Response &response) {
		return place();
	}

	bool place() {

		group->place(object_to_manipulate);

		return true;

		std::vector<manipulation_msgs::PlaceLocation> loc;

		geometry_msgs::PoseStamped p;
		/*
		 p.header.frame_id = BASE_LINK;
		 p.pose.position = object_to_manipulate_position.point;

		 p.pose.orientation.x = 0;
		 p.pose.orientation.y = 0;
		 p.pose.orientation.z = 0;
		 p.pose.orientation.w = 1;

		 */
		//p = pickup_grasp.grasp_pose;
		make_pose_reachable_by_5DOF_katana(p);

		ROS_DEBUG_STREAM("Place Pose: " << p.pose);

		manipulation_msgs::PlaceLocation g;
		g.place_pose = p;

		g.approach.direction.vector.z = -1.0;
		g.retreat.direction.vector.x = -1.0;
		g.retreat.direction.header.frame_id = BASE_LINK;
		g.approach.direction.header.frame_id = GRIPPER_FRAME;
		g.approach.min_distance = 0.1;
		g.approach.desired_distance = 0.2;
		g.retreat.min_distance = 0.1;
		g.retreat.desired_distance = 0.25;

		g.post_place_posture.name.resize(1, FINGER_JOINT);
		g.post_place_posture.position.resize(1);
		g.post_place_posture.position[0] = 0.30;

		loc.push_back(g);

		//group->setSupportSurfaceName("table");

		/* Option path constraints (e.g. to always stay upright)
		 // add path constraints
		 moveit_msgs::Constraints constr;
		 constr.orientation_constraints.resize(1);
		 moveit_msgs::OrientationConstraint &ocm = constr.orientation_constraints[0];
		 ocm.link_name = "r_wrist_roll_link";
		 ocm.header.frame_id = p.header.frame_id;
		 ocm.orientation.x = 0.0;
		 ocm.orientation.y = 0.0;
		 ocm.orientation.z = 0.0;
		 ocm.orientation.w = 1.0;
		 ocm.absolute_x_axis_tolerance = 0.2;
		 ocm.absolute_y_axis_tolerance = 0.2;
		 ocm.absolute_z_axis_tolerance = M_PI;
		 ocm.weight = 1.0;
		 group->setPathConstraints(constr);
		 group->setPlannerId("RRTConnectkConfigDefault");
		 */

		group->place(object_to_manipulate, loc);

		/*
		 //create a place location
		 //geometry_msgs::PoseStamped place_location = pickup_location;

		 geometry_msgs::PoseStamped place_location;

		 place_location.header.stamp = ros::Time::now();

		 // ----- put the object down
		 ROS_INFO("Calling the place action");
		 object_manipulation_msgs::PlaceGoal place_goal;

		 place_location.header.frame_id = KATANA_BASE_LINK;



		 place_location.pose.position.x =
		 normalPoseRobotFrame.pose.position.x;
		 place_location.pose.position.y =
		 normalPoseRobotFrame.pose.position.y;
		 place_location.pose.position.z =
		 pickup_result.grasp.grasp_pose.position.z; //TODO: to calculate z object higth needs to be considered

		 tf::Transform position;
		 tf::poseMsgToTF(place_location.pose, position);

		 tf::Vector3 shift_direction;
		 shift_direction.setX(place_location.pose.position.x);
		 shift_direction.setY(place_location.pose.position.y);
		 shift_direction.setZ(0);
		 shift_direction.normalize();

		 tf::Vector3 shift_direction_inverse;
		 shift_direction_inverse.setX(-shift_direction.getX());
		 shift_direction_inverse.setY(-shift_direction.getY());
		 shift_direction_inverse.setZ(-shift_direction.getZ());

		 //transform place_position to the start of the test line
		 tf::Transform trans(tf::Quaternion(0, 0, 0, 1.0),
		 shift_direction_inverse * place_position_tolerance_in_meter);

		 position = trans * position;

		 //move along the test line and add positions to place_goal.place_locations
		 for (float offset = 0; offset <= place_position_tolerance_in_meter;
		 offset += place_planner_step_size_in_meter) {

		 tf::Transform trans(tf::Quaternion(0, 0, 0, 1.0),
		 shift_direction * place_planner_step_size_in_meter);

		 position = trans * position;

		 geometry_msgs::Pose position_pose;
		 tf::poseTFToMsg(position, position_pose);

		 place_location.pose = position_pose;

		 // Convert quaternion to RPY.
		 tf::Quaternion q;
		 double roll, pitch, yaw;
		 tf::quaternionMsgToTF(pickup_result.grasp.grasp_pose.orientation,
		 q);
		 tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

		 //determine yaw which is compatible with the Katana 300 180 kinematics.
		 yaw = atan2(place_location.pose.position.y,
		 place_location.pose.position.x);

		 tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, pitch, yaw);

		 place_location.pose.orientation.x = quat.getX();
		 place_location.pose.orientation.y = quat.getY();
		 place_location.pose.orientation.z = quat.getZ();
		 place_location.pose.orientation.w = quat.getW();

		 ROS_INFO(
		 "Place Location: XYZ, Quaternions xyzw = %lf,%lf,%lf, (%lf, %lf, %lf, %lf)", place_location.pose.position.x, place_location.pose.position.y, place_location.pose.position.z, place_location.pose.orientation.x, place_location.pose.orientation.y, place_location.pose.orientation.z, place_location.pose.orientation.w);

		 place_goal.place_locations.push_back(place_location);

		 }

		 //end place planner

		 //the collision names of both the objects and the table
		 //same as in the pickup action
		 place_goal.collision_object_name =
		 processing_call.response.collision_object_names.at(
		 object_to_pick_ind);
		 place_goal.collision_support_surface_name =
		 processing_call.response.collision_support_surface_name;

		 //set grasp orientation to identity (all info is already given by place_location)

		 place_goal.grasp.grasp_pose.orientation.x = 0;
		 place_goal.grasp.grasp_pose.orientation.y = 0;
		 place_goal.grasp.grasp_pose.orientation.z = 0;
		 place_goal.grasp.grasp_pose.orientation.w = 1;
		 place_goal.grasp.grasp_pose.position.x = 0;
		 place_goal.grasp.grasp_pose.position.y = 0;
		 place_goal.grasp.grasp_pose.position.z = 0;

		 //use the arm to place
		 place_goal.arm_name = "arm";
		 //padding used when determining if the requested place location
		 //would bring the object in collision with the environment
		 place_goal.place_padding = 0.02;
		 //how much the gripper should retreat after placing the object
		 place_goal.desired_retreat_distance = 0.06;
		 place_goal.min_retreat_distance = 0.05;
		 //we will be putting down the object along the "vertical" direction
		 //which is along the z axis in the base_link frame
		 geometry_msgs::Vector3Stamped direction;
		 direction.header.stamp = ros::Time::now();
		 direction.header.frame_id = KATANA_BASE_LINK;
		 direction.vector.x = 0;
		 direction.vector.y = 0;
		 direction.vector.z = -1;
		 place_goal.approach.direction = direction;
		 //request a vertical put down motion of 10cm before placing the object
		 place_goal.approach.desired_distance = 0.06;
		 place_goal.approach.min_distance = 0.05;
		 //we are not using tactile based placing
		 place_goal.use_reactive_place = false;
		 //send the goal
		 place_client->sendGoal(place_goal);
		 while (!place_client->waitForResult(ros::Duration(10.0))) {
		 ROS_INFO("Waiting for the place action...");
		 if (!nh.ok())
		 return -1;
		 }
		 object_manipulation_msgs::PlaceResult place_result =
		 *(place_client->getResult());
		 if (place_client->getState()
		 != actionlib::SimpleClientGoalState::SUCCEEDED) {
		 if (place_result.manipulation_result.value
		 == object_manipulation_msgs::ManipulationResult::RETREAT_FAILED) {
		 ROS_WARN(
		 "Place failed with error RETREAT_FAILED, ignoring! This may lead to collision with the object we just placed!");
		 } else {
		 ROS_ERROR(
		 "Place failed with error code %d", place_result.manipulation_result.value);
		 return -1;
		 }

		 }

		 //success!
		 ROS_INFO("Success! Object moved.");

		 move_arm_out_of_the_way();

		 object_in_gripper = 0;
		 */

		return true;
	}

	bool move_arm_out_of_the_way_callback(std_srvs::Empty::Request &request,
			std_srvs::Empty::Response &response) {
		if (move_arm_out_of_the_way()) {
			return false;
		}
		return true;
	}

	int move_arm_out_of_the_way() {

		//set_joint_goal();

		//clear_collision_map();

		ROS_INFO("Move arm out of the way.");

		//move arm to initial home state as defined in the urdf
		group->setNamedTarget("home");

		group->asyncMove();

		//clear_collision_map();

		ROS_INFO("Arm moved out of the way.");

		return 1;
	}

	int clear_collision_map() {

		//if (sim)
		ros::Duration(2.0).sleep();	// only necessary for Gazebo (the simulated Kinect point cloud lags, so we need to wait for it to settle)

		// ----- reset collision map
		ROS_INFO("Clearing collision map");
		std_srvs::Empty empty;
		if (!collider_reset_srv.call(empty)) {
			ROS_ERROR("Collider reset service failed");
			return -1;
		}
		//if (sim)
		ros::Duration(3.0).sleep();	// wait for collision map to be completely cleared

		return 0;

	}

	void receive_clicked_point_CB(
			const geometry_msgs::PointStamped::ConstPtr& msg) {
		ROS_INFO(
				"Point received: x: %f, y: %f, z: %f ", msg->point.x, msg->point.y, msg->point.z);

		/*
		 //Throw away old received clicked points
		 if ((ros::Time::now().sec - msg->header.stamp.sec)
		 > message_receive_dead_time_in_sec) {
		 return;
		 }
		 */

		set_pickup_point(*msg);

		clicked = true;

		ROS_INFO("Clicked!!!!");

		determine_normal_of_point_cloud_at_clicked_point();

		pickup_nearest_segmented_object();

	}

	bool determine_normal_of_point_cloud_at_clicked_point() {
		visualization_msgs::Marker marker;

		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(pcl_input_point_cloud);

		//transform selected point from robot frame (BASE_LINK) to Kinect frame (/kinect_rgb_optical_frame)
		tf::Stamped<tf::Vector3> searchPointInRobotFrame;

		tf::pointStampedMsgToTF(desired_pickup_point, searchPointInRobotFrame);

		tf::StampedTransform transformRobotToPointCloud;

		try {
			tf_listener->lookupTransform(pcl_input_point_cloud->header.frame_id,
					BASE_LINK, ros::Time(0), transformRobotToPointCloud);
		} catch (tf::TransformException& ex) {
			ROS_ERROR("%s", ex.what());
		}

		tf::Vector3 searchPointPointCloudFrame = transformRobotToPointCloud
				* searchPointInRobotFrame;

		pcl::PointXYZ searchPoint;

		searchPoint.x = searchPointPointCloudFrame.getX();
		searchPoint.y = searchPointPointCloudFrame.getY();
		searchPoint.z = searchPointPointCloudFrame.getZ();

		float radius = 0.02;

		ROS_INFO(
				"Search searchPointWorldFrame set to: x: %f, y: %f, z: %f ", searchPoint.x, searchPoint.y, searchPoint.z);

		// Neighbors within radius search

		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;

		ROS_DEBUG_STREAM(
				"Input cloud frame id: " << pcl_input_point_cloud->header.frame_id);

		if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
				pointRadiusSquaredDistance) > 5) {
			for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
				//ROS_DEBUG_STREAM(
				//		"   " << cloud->points[pointIdxRadiusSearch[i]].x << " " << cloud->points[pointIdxRadiusSearch[i]].y << " " << cloud->points[pointIdxRadiusSearch[i]].z << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl);

				marker.pose.position.x =
						pcl_input_point_cloud->points[pointIdxRadiusSearch[i]].x;
				marker.pose.position.y =
						pcl_input_point_cloud->points[pointIdxRadiusSearch[i]].y;
				marker.pose.position.z =
						pcl_input_point_cloud->points[pointIdxRadiusSearch[i]].z;

				//show markers in kinect frame
				marker.header.frame_id = pcl_input_point_cloud->header.frame_id;
				marker.id = i;
				marker.ns = "selection";
				marker.header.stamp = ros::Time::now();
				marker.action = visualization_msgs::Marker::ADD;
				marker.lifetime = ros::Duration();
				marker.type = visualization_msgs::Marker::SPHERE;
				marker.scale.x = 0.003;
				marker.scale.y = 0.003;
				marker.scale.z = 0.003;
				marker.color.r = 1;
				marker.color.g = 0;
				marker.color.b = 0;
				marker.color.a = 1.0;
				vis_marker_publisher.publish(marker);

			}

			ROS_INFO_STREAM(
					"Number of points in radius: " << pointIdxRadiusSearch.size());

			//Determine Normal from points in radius

			Eigen::Vector4f plane_parameters;

			float curvature;

			pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimator;

			normalEstimator.computePointNormal(*pcl_input_point_cloud,
					pointIdxRadiusSearch, plane_parameters, curvature);

			normal.getVector4fMap() = plane_parameters;

			ROS_INFO(
					"Normal: x: %f, y: %f, z: %f ", normal.x, normal.y, normal.z);
			ROS_DEBUG_STREAM("Normal: " << normal);

			/*
			 tf::Vector3 normalTF(normal.x, normal.y, normal.z);

			 normalTF = transformWorldToPointCloud.inverse() * normalTF;
			 */
			geometry_msgs::PoseStamped normalPosePointCloudFrame;

			normalPosePointCloudFrame.header.frame_id =
					pcl_input_point_cloud->header.frame_id;

			normalPosePointCloudFrame.pose.position.x = searchPoint.x;
			normalPosePointCloudFrame.pose.position.y = searchPoint.y;
			normalPosePointCloudFrame.pose.position.z = searchPoint.z;

			//determine orientation of normal for marker

			btVector3 axis(normal.x, normal.y, normal.z);
			//tf::Vector3 axis(normal.x, normal.y, normal.z);

			btVector3 marker_axis(1, 0, 0);
			//tf::Vector3 marker_axis(1,0,0);

			btQuaternion qt(marker_axis.cross(axis.normalize()),
					marker_axis.angle(axis.normalize()));

			qt.normalize();

			//tf::Quaternion qt2(marker_axis.cross(axis.normalize()),
			//	marker_axis.angle(axis.normalize()));

			/*
			 geometry_msgs::Quaternion quat_msg;
			 tf::quaternionTFToMsg(qt, quat_msg);
			 normalPosePointCloudFrame.pose.orientation = quat_msg;
			 */

			normalPosePointCloudFrame.pose.orientation.x = qt.getX();
			normalPosePointCloudFrame.pose.orientation.y = qt.getY();
			normalPosePointCloudFrame.pose.orientation.z = qt.getZ();
			normalPosePointCloudFrame.pose.orientation.w = qt.getW();

			ROS_DEBUG_STREAM(
					"Pose in Point Cloud Frame: " << normalPosePointCloudFrame.pose);

			//transform normal pose to Katana base

			try {
				tf_listener->transformPose(BASE_LINK, normalPosePointCloudFrame,
						normalPoseRobotFrame);
			} catch (const tf::TransformException &ex) {

				ROS_ERROR("%s", ex.what());

			} catch (const std::exception &ex) {

				ROS_ERROR("%s", ex.what());

			}

			ROS_DEBUG_STREAM(
					"base link frame frame id: " << normalPoseRobotFrame.header.frame_id);

			marker.pose = normalPoseRobotFrame.pose;

			//marker.pose = normalPose.pose;
			marker.header.frame_id = BASE_LINK;
			marker.id = 12345;
			marker.ns = "normal";
			marker.header.stamp = ros::Time::now();
			marker.action = visualization_msgs::Marker::ADD;
			marker.lifetime = ros::Duration();
			marker.type = visualization_msgs::Marker::ARROW;
			marker.scale.x = 0.05;
			marker.scale.y = 0.005;
			marker.scale.z = 0.005;
			marker.color.r = 1;
			marker.color.g = 0;
			marker.color.b = 0;
			marker.color.a = 1.0;
			vis_marker_publisher.publish(marker);

			ROS_DEBUG_STREAM(
					"Nomal pose in base link frame: " << normalPoseRobotFrame.pose);

			return true;

		} else {

			ROS_ERROR(
					"Normal:No Points found inside search radius! Search radios too small?");
			return false;

		}

	}

	void receive_cloud_CB(
			const sensor_msgs::PointCloud2ConstPtr& ros_input_cloud) {

		//if (clicked) {
		//clicked = false;
		//ROS_INFO("Received point cloud");
		pcl::fromROSMsg(*ros_input_cloud, *pcl_input_point_cloud);
		//}

	}

	bool find_nearest_object() {

		//desired_pickup_point



		moveit_msgs::GetPlanningScene get_planning_scene_call;

		//get all planning scene objects
		get_planning_scene_call.request.components.components =
				moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY;

		if (!get_planning_scene_srv.call(get_planning_scene_call)) {
			ROS_ERROR("Get Planning Scene call failed");
			return false;
		}

		if (get_planning_scene_call.response.scene.world.collision_objects.empty()) {
			ROS_ERROR("Get Planning scene returned nothing");
			return false;
		}

		geometry_msgs::PointStamped point;

		// convert point to base_link frame
		tf_listener->transformPoint("/base_link", desired_pickup_point, point);

		ROS_DEBUG_STREAM(
				"Pickup Point test which object is nearest: " << point.point.x << " " << point.point.y << " " << point.point.z);

		// find the closest object
		double nearest_dist = 1e6;
		int nearest_object_ind = -1;

		int number_of_scene_objects =
				get_planning_scene_call.response.scene.world.collision_objects.size()
						- 1;

		ROS_INFO_STREAM("Number of Scene Objects: " << number_of_scene_objects);

		for (int i = 0; i < number_of_scene_objects; i++) {

			geometry_msgs::Point object_position_in_base_link_frame =
					get_planning_scene_call.response.scene.world.collision_objects[i].primitive_poses[0].position;
			ROS_INFO_STREAM(
					"object " << i << " position: " << object_position_in_base_link_frame);
			/*
			 geometry_msgs::PointStamped object_position_in_base_link_frame;
			 tf_listener->transformPoint("/base_link", object_positions[i],
			 object_position_in_base_link_frame);
			 */
			double dist = sqrt(
					pow(object_position_in_base_link_frame.x - point.point.x,
							2.0)
							+ pow(
									object_position_in_base_link_frame.y
											- point.point.y, 2.0)
							+ pow(
									object_position_in_base_link_frame.z
											- point.point.z, 2.0));
			if (dist < nearest_dist) {
				nearest_dist = dist;
				nearest_object_ind = i;
				object_to_manipulate_position = geometry_msgs::Point(
						object_position_in_base_link_frame);
			}
		}

		if (nearest_object_ind > -1) {
			ROS_INFO("NEAREST");
			ROS_INFO(
					"nearest object ind: %d (distance: %f", nearest_object_ind, nearest_dist);

			//object_to_manipulate_position = get_planning_scene_call.response.scene.world.collision_objects[nearest_object_ind].primitive_poses[0].position;

			ROS_INFO_STREAM(
					"Object Position: " << object_to_manipulate_position);

			object_to_manipulate_index = nearest_object_ind;

			string id =
					get_planning_scene_call.response.scene.world.collision_objects[nearest_object_ind].id;

			object_to_manipulate_object = get_planning_scene_call.response.scene.world.collision_objects[nearest_object_ind];

			object_to_manipulate = id.c_str();

			return true;

		} else {
			ROS_ERROR("No nearby objects. Unable to select a pickup target");
			return false;
		}

	}

	void remove_all_scene_objects(){

		moveit_msgs::GetPlanningScene get_planning_scene_call;

		//get all planning scene objects
		get_planning_scene_call.request.components.components =
				moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY;

		if (!get_planning_scene_srv.call(get_planning_scene_call)) {
			ROS_ERROR("Get Planning Scene call failed");
			return;
		}

		if (get_planning_scene_call.response.scene.world.collision_objects.empty()) {
			ROS_ERROR("Get Planning scene returned nothing");
			return;
		}

		int number_of_scene_objects =
				get_planning_scene_call.response.scene.world.collision_objects.size();


		for (int i = 0; i < number_of_scene_objects; i++) {

			moveit_msgs::CollisionObject collision_object;

			collision_object.header.stamp = ros::Time::now();
			collision_object.header.frame_id = BASE_LINK;

			collision_object.id = get_planning_scene_call.response.scene.world.collision_objects[i].id;

			collision_object.operation = moveit_msgs::CollisionObject::REMOVE;

			pub_collision_object.publish(collision_object);
		}
	}

	Marker makeBox(InteractiveMarker &msg) {

		//TODO: make gripper nicer http://answers.ros.org/question/12840/drawing-the-pr2-gripper-in-rviz/
		//https://github.com/ros-interactive-manipulation/pr2_object_manipulation/tree/groovy-devel/manipulation/pr2_marker_control/src
		Marker marker;

		marker.type = Marker::CUBE;
		marker.scale.x = 0.03;
		marker.scale.y = 0.1;
		marker.scale.z = 0.01;
		marker.color.r = 0.5;
		marker.color.g = 0.5;
		marker.color.b = 0.5;
		marker.color.a = 0.5;

		return marker;
	}

	Marker makeBox6DOF(InteractiveMarker &msg) {

		//TODO: make gripper nicer http://answers.ros.org/question/12840/drawing-the-pr2-gripper-in-rviz/
		//https://github.com/ros-interactive-manipulation/pr2_object_manipulation/tree/groovy-devel/manipulation/pr2_marker_control/src
		Marker marker;

		marker.type = Marker::CUBE;
		marker.scale.x = 0.03;
		marker.scale.y = 0.1;
		marker.scale.z = 0.01;
		marker.color.r = 0;
		marker.color.g = 1;
		marker.color.b = 0;
		marker.color.a = 0.9;

		return marker;
	}

	InteractiveMarkerControl& makeBoxControl(InteractiveMarker &msg) {
		InteractiveMarkerControl control;
		control.always_visible = true;
		control.markers.push_back(makeBox6DOF(msg));
		msg.controls.push_back(control);

		return msg.controls.back();
	}

	void makeSelectableMarker(int id, geometry_msgs::PoseStamped pose){

		InteractiveMarker int_marker;
		int_marker.header.frame_id = BASE_LINK;

		int_marker.pose = pose.pose;
		int_marker.scale = 0.1;

		InteractiveMarkerControl control;
		control.interaction_mode = visualization_msgs::InteractiveMarkerControl::BUTTON;
		control.markers.push_back(makeBox(int_marker));

		int_marker.controls.push_back(control);

		int_marker.name = "selectable_" + boost::lexical_cast<std::string>(id);
		int_marker.description = "Selectable marker";

		ROS_INFO_STREAM("Created marker: " << int_marker.name);

		server->insert(int_marker);


		server->setCallback(int_marker.name,
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));

		server->applyChanges();
	}

	void setSelectableMarkerToGreen(int id){

		InteractiveMarker int_marker;

		server->get("selectable_" + boost::lexical_cast<std::string>(id),int_marker);

		int_marker.controls[0].markers[0].color.r = 0;
		int_marker.controls[0].markers[0].color.g = 1;;
		int_marker.controls[0].markers[0].color.b = 0;
		int_marker.controls[0].markers[0].color.a = 0.9;

		server->insert(int_marker);

		server->applyChanges();
	}

	void setSelectableMarkerToRed(int id){

		InteractiveMarker int_marker;

		server->get("selectable_" + boost::lexical_cast<std::string>(id),int_marker);

		int_marker.controls[0].markers[0].color.r = 1;
		int_marker.controls[0].markers[0].color.g = 0;;
		int_marker.controls[0].markers[0].color.b = 0;
		int_marker.controls[0].markers[0].color.a = 0.9;

		server->insert(int_marker);

		server->applyChanges();
	}

	void make6DofMarker() {

		//menu
		menu_handler.insert("Grasp",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("Grasp Manually",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("Pickup Plan only",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("Reachability Test own pick function",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("Reachability Test moveit pick",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("Reachability Test just move to",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));


		InteractiveMarker int_marker;
		int_marker.header.frame_id = BASE_LINK;


		int_marker.scale = 0.1;

		int_marker.name = "simple_6dof";
		int_marker.description = "Simple 6-DOF Control";

		// insert a box
		makeBoxControl(int_marker);
		int_marker.controls[0].interaction_mode =
				visualization_msgs::InteractiveMarkerControl::MOVE_ROTATE_3D;

		InteractiveMarkerControl control;

		control.orientation.w = 1;
		control.orientation.x = 1;
		control.orientation.y = 0;
		control.orientation.z = 0;
		control.name = "rotate_x";
		control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
		int_marker.controls.push_back(control);
		control.name = "move_x";
		control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
		int_marker.controls.push_back(control);

		control.orientation.w = 1;
		control.orientation.x = 0;
		control.orientation.y = 1;
		control.orientation.z = 0;
		control.name = "rotate_z";
		control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
		int_marker.controls.push_back(control);
		control.name = "move_z";
		control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
		int_marker.controls.push_back(control);

		control.orientation.w = 1;
		control.orientation.x = 0;
		control.orientation.y = 0;
		control.orientation.z = 1;
		control.name = "rotate_y";
		control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
		int_marker.controls.push_back(control);
		control.name = "move_y";
		control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
		int_marker.controls.push_back(control);

		server->insert(int_marker);

		server->setCallback(int_marker.name,
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));

		menu_handler.apply(*server, int_marker.name);

		server->applyChanges();

		ROS_INFO("Interactive marker created");
	}

	void setMarkerPoseToFirstGrasp() {

		setMarkerToPoseStamped(pickup_grasps[0].grasp_pose);

	}

	void setMarkerToPoseStamped(geometry_msgs::PoseStamped pose){

		InteractiveMarker int_marker;
		server->get("simple_6dof", int_marker);

		int_marker.pose = pose.pose;

		currentMarkerPose = pose;

		server->insert(int_marker);

		server->applyChanges();

	}

	void processFeedback(
			const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback) {
		std::ostringstream s;
		s << "Feedback from marker '" << feedback->marker_name << "' "
				<< " / control '" << feedback->control_name << "'";

		std::ostringstream mouse_point_ss;
		if (feedback->mouse_point_valid) {
			mouse_point_ss << " at " << feedback->mouse_point.x << ", "
					<< feedback->mouse_point.y << ", "
					<< feedback->mouse_point.z << " in frame "
					<< feedback->header.frame_id;
		}

		switch (feedback->event_type) {
		case visualization_msgs::InteractiveMarkerFeedback::BUTTON_CLICK:
			ROS_INFO_STREAM(
					s.str() << ": button click" << mouse_point_ss.str() << ".");
			break;

		case visualization_msgs::InteractiveMarkerFeedback::MENU_SELECT:
			ROS_INFO_STREAM(
					s.str() << ": menu item " << feedback->menu_entry_id << " clicked" << mouse_point_ss.str() << ".");

			switch (feedback->menu_entry_id) {

			//"Grasp"
			case 1:
				currentMarkerPose.pose = feedback->pose;
				pickup_with_current_marker_pose();
				break;

			//Pickup manually
			case 2:
				pickup_manually();
				break;

			//"Plan only"
			case 3:
				pickup_plan_only();
				break;
			//Reachability test
			case 4:
				do_reachability_test(2);
				break;

			case 5:

				do_reachability_test(1);
				break;

			case 6:
				//do test 1 and 2 each 3 times
				//for(int i = 0 ; i<3 ; i++){
					do_reachability_test(0);
				//}

				break;
			}
			break;

		case visualization_msgs::InteractiveMarkerFeedback::POSE_UPDATE:
			ROS_INFO_STREAM(
					s.str() << ": pose changed" << "\nposition = " << feedback->pose.position.x << ", " << feedback->pose.position.y << ", " << feedback->pose.position.z << "\norientation = " << feedback->pose.orientation.w << ", " << feedback->pose.orientation.x << ", " << feedback->pose.orientation.y << ", " << feedback->pose.orientation.z << "\nframe: " << feedback->header.frame_id << " time: " << feedback->header.stamp.sec << "sec, " << feedback->header.stamp.nsec << " nsec");

			currentMarkerPose.pose.orientation = feedback->pose.orientation;
			currentMarkerPose.pose.position = feedback->pose.position;
			currentMarkerPose.header.frame_id = feedback->header.frame_id;
			setMarkerToPoseStamped(currentMarkerPose);

			break;

		case visualization_msgs::InteractiveMarkerFeedback::MOUSE_DOWN:
			ROS_INFO_STREAM(
					s.str() << ": mouse down" << mouse_point_ss.str() << ".");
			break;

		case visualization_msgs::InteractiveMarkerFeedback::MOUSE_UP:
			ROS_INFO_STREAM(
					s.str() << ": mouse up" << mouse_point_ss.str() << ".");
			break;
		}

		server->applyChanges();
	}

};

int main(int argc, char **argv) {
	//initialize the ROS node
	ros::init(argc, argv, "pick_and_place_app");

	ros::NodeHandle nh;

	ros::CallbackQueue clicks_queue;

	Pick_and_place_app *app = new Pick_and_place_app(&nh);

	//Advertise pickup_object Service and use async callback queue.
	ros::AdvertiseServiceOptions advertiseServiceOptions =
			ros::AdvertiseServiceOptions::create<std_srvs::Empty>(
					"pickup_object",
					boost::bind(&Pick_and_place_app::pickup_callback, app, _1,
							_2), ros::VoidPtr(), &clicks_queue);
	ros::ServiceServer pickup_object_srv = nh.advertiseService(
			advertiseServiceOptions);

	//Advertise place_object Service and use async callback queue.
	advertiseServiceOptions = ros::AdvertiseServiceOptions::create<
			std_srvs::Empty>("place_object",
			boost::bind(&Pick_and_place_app::place_callback, app, _1, _2),
			ros::VoidPtr(), &clicks_queue);
	ros::ServiceServer place_object_srv = nh.advertiseService(
			advertiseServiceOptions);

	//Advertise "move_arm_out_of_the_way" Service and use async callback queue.
	advertiseServiceOptions = ros::AdvertiseServiceOptions::create<
			std_srvs::Empty>("move_arm_out_of_the_way",
			boost::bind(&Pick_and_place_app::move_arm_out_of_the_way_callback,
					app, _1, _2), ros::VoidPtr(), &clicks_queue);
	ros::ServiceServer move_arm_out_of_the_way_srv = nh.advertiseService(
			advertiseServiceOptions);

	// Create a ROS subscriber for the input point cloud

	ros::Subscriber point_cloud_subscriber = nh.subscribe(
			"/kinect/depth_registered/points", 1,
			&Pick_and_place_app::receive_cloud_CB, app);

	//Async Queue for Clicks_queue, because the moveit functions like pickup, move don't return in synchronous callbacks
	ros::SubscribeOptions options = ros::SubscribeOptions::create<
			geometry_msgs::PointStamped>("/clicked_point", 1,
			boost::bind(&Pick_and_place_app::receive_clicked_point_CB, app, _1),
			ros::VoidPtr(), &clicks_queue);

	ros::Subscriber rviz_click_subscriber = nh.subscribe(options);

	ROS_INFO("Subscriped to point cloud and clicked_point");

	ROS_INFO("Pick and Place v 0.2.1 ready to take commands.");

	//Async Spinner for Clicks_queue, because the moveit functions like pickup, move don't return in synchronous callbacks
	ros::AsyncSpinner spinner(0, &clicks_queue);
	spinner.start();

	ros::spin();

	return 0;
}

