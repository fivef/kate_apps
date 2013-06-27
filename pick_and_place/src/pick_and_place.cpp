#include <ros/ros.h>

#include <tf/transform_listener.h>
#include <actionlib/client/simple_action_client.h>

#include <std_srvs/Empty.h>
#include "geometry_msgs/Point.h"
#include <boost/spirit/include/classic.hpp>
#include "visualization_msgs/Marker.h"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <math.h>
#include <sstream>

#include <object_manipulation_msgs/FindClusterBoundingBox2.h>
#include <tabletop_object_detector/TabletopDetection.h>
#include "/home/sp/groovy_rosbuild_workspace/overlay/tabletop_collision_map_processing/srv_gen/cpp/include/tabletop_collision_map_processing/TabletopCollisionMapProcessing.h"

//for PTU dynamic reconfigure
#include <dynamic_reconfigure/DoubleParameter.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>

// MoveIt!
#include <moveit/pick_place/pick_place.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/move_group_interface/move_group.h>
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
const std::string PICKUP_ACTION_NAME =
		"/object_manipulator/object_manipulator_pickup";
const std::string PLACE_ACTION_NAME =
		"/object_manipulator/object_manipulator_place";
const std::string MOVE_ARM_ACTION_NAME = "/move_arm";
const std::string COLLIDER_RESET_SERVICE_NAME = "/collider_node/reset";

static const std::string CLUSTER_BOUNDING_BOX2_3D_NAME = "find_cluster_bounding_box2_3d";

//! General base class for all exceptions originating in the collision map interface
class CollisionMapException : public std::runtime_error
{
 public:
 CollisionMapException(const std::string error) : std::runtime_error("collision map: "+error) {};
};

class Pick_and_place_app {

private:
	static const int message_receive_dead_time_in_sec = 5;

	//1 normal directly selects grasp pose / 0 nearest objects to selected point is grasped
	int DIRECTLY_SELECT_GRASP_POSE;

	static const float place_position_tolerance_in_meter = 0.03;
	static const float place_planner_step_size_in_meter = 0.005;

	//the distance in y direction from the position where the object was picked up to the place position
	static const float place_offset = 0.20;

	static const size_t NUM_JOINTS = 5;

	int object_in_gripper;

	std::string KATANA_BASE_LINK;

	std::string BASE_FOOTPRINT;

	std::string WRIST_JOINT;

	std::string FINGER_JOINT;

	std::string ARM_NAME;

	static const int NORMAL = 1;
	static const int SEGMENTED_OBJECT_SELECTION = 2;
	std::string WORKING_MODE;

	geometry_msgs::Point desired_pickup_point;

	int running;

	ros::Publisher vis_marker_publisher;

	tf::TransformListener *tf_listener;

	geometry_msgs::PoseStamped normalPoseRobotFrame;

	pcl::PointXYZ normal;

	move_group_interface::MoveGroup *group;


	ros::Publisher pub_collision_object;

	//only calculate normal once after click
	int clicked;

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

	ros::ServiceServer pickup_object_srv;
	ros::ServiceServer place_object_srv;
	ros::ServiceServer move_arm_out_of_the_way_srv;

	ros::NodeHandle nh;


public:

	Pick_and_place_app(ros::NodeHandle *_nh) {

		nh = *_nh;


		//get parameters from parameter server
		nh.param<int>("sim", sim, 0);

		nh.param<int>("DIRECTLY_SELECT_GRASP_POSE", DIRECTLY_SELECT_GRASP_POSE, 1);

		nh.param<std::string>("KATANA_BASE_LINK", KATANA_BASE_LINK,"/katana_base_link");

		//nh.param<std::string>("BASE_FOOTPRINT", BASE_FOOTPRINT,"/base_footprint"); //TODO change back
		nh.param<std::string>("BASE_FOOTPRINT", BASE_FOOTPRINT,"/base_link");

		nh.param<std::string>("WRIST_JOINT", WRIST_JOINT,"/katana_gripper_link");
		nh.param<std::string>("FINGER_JOINT", FINGER_JOINT,"/katana_l_finger_joint");
		nh.param<std::string>("ARM_NAME", ARM_NAME,"arm");

		clicked = 0;

		object_in_gripper = 0;


		// create TF listener
		tf_listener = new tf::TransformListener();


		pub_collision_object = nh.advertise<moveit_msgs::CollisionObject>("collision_object", 10);

		ros::WallDuration(1.0).sleep();


		group = new move_group_interface::MoveGroup(ARM_NAME);

		//group->setPlanningTime(5.0);

		group->setGoalTolerance(0.01);
		//group->setGoalOrientationTolerance(1.0);


		//group->allowReplanning(true);



		pickup_object_srv = nh.advertiseService("pickup_object",
				&Pick_and_place_app::pickup_callback, this);

		place_object_srv = nh.advertiseService("place_object",
				&Pick_and_place_app::place_callback, this);
		move_arm_out_of_the_way_srv = nh.advertiseService(
				"move_arm_out_of_the_way",
				&Pick_and_place_app::move_arm_out_of_the_way_callback, this);

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


		vis_marker_publisher = nh.advertise<visualization_msgs::Marker>(
				"pick_and_place_markers", 128);

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

		cluster_bounding_box2_3d_client_ = nh.serviceClient<object_manipulation_msgs::FindClusterBoundingBox2>
		    (CLUSTER_BOUNDING_BOX2_3D_NAME, true);


		//Sets the kinects tilt angle

		set_kinect_ptu("kurtana_pitch_joint", 0.85);


		ROS_INFO("Kinect lined up.");


		//TODO: only move arm out of the way in gazebo simulation because the real robot already is in init position
		move_arm_out_of_the_way();


	}

	~Pick_and_place_app() {

		//Sets the kinects tilt angle
		set_kinect_ptu("kurtana_pitch_joint", 0.0);

	}

	void set_pickup_point(geometry_msgs::Point point) {

		desired_pickup_point.x = point.x;
		desired_pickup_point.y = point.y;
		desired_pickup_point.z = point.z;

		ROS_INFO_STREAM(
				"Pickup Point set to: " << desired_pickup_point.x << " " << desired_pickup_point.y << " " << desired_pickup_point.z);

	}

	void set_kinect_ptu(std::string joint_name, double value){

		dynamic_reconfigure::ReconfigureRequest srv_req;
			dynamic_reconfigure::ReconfigureResponse srv_resp;
			dynamic_reconfigure::DoubleParameter double_param;
			dynamic_reconfigure::Config conf;

			double_param.name = joint_name;
			double_param.value = value;
			conf.doubles.push_back(double_param);

			srv_req.config = conf;

			ros::service::call("/joint_commander/set_parameters", srv_req, srv_resp);
	}

	geometry_msgs::Point get_pickup_point() {

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

	int pickup() {
/*
		// ----- pick up object near point (in meter) relative to base_footprint
		geometry_msgs::PointStamped pickup_point;
		pickup_point.header.frame_id = KURTANA_BASE_LINK;
		pickup_point.point.x = desired_pickup_point.x;
		pickup_point.point.y = desired_pickup_point.y;
		pickup_point.point.z = desired_pickup_point.z;
*/
		/*
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


		ROS_INFO("Add table to planning scene");

		moveit_msgs::CollisionObject collision_object;

		collision_object.header.stamp = ros::Time::now();
		collision_object.header.frame_id = detection_call.response.detection.table.pose.header.frame_id;

		ROS_INFO_STREAM("Collision Object frame: " << detection_call.response.detection.table.pose.header.frame_id);

		//add table to planning scene

		// remove table
		collision_object.id = "table";
		collision_object.operation = moveit_msgs::CollisionObject::REMOVE;
		pub_collision_object.publish(collision_object);

		// add table
		collision_object.operation = moveit_msgs::CollisionObject::ADD;


		collision_object.meshes.resize(1);
		collision_object.meshes[0] = detection_call.response.detection.table.convex_hull;
		collision_object.mesh_poses.resize(1);
		collision_object.mesh_poses[0] = detection_call.response.detection.table.pose.pose;

		pub_collision_object.publish(collision_object);

		//ROS_DEBUG_STREAM("Table: " << detection_call.response.detection.table);

		ROS_INFO_STREAM("Add clusters");

		//add objects to planning scene
		for(unsigned int i = 0; i < detection_call.response.detection.clusters.size(); i++){

			sensor_msgs::PointCloud2 pc2;
			sensor_msgs::convertPointCloudToPointCloud2(detection_call.response.detection.clusters[i],pc2);
			geometry_msgs::PoseStamped poseStamped;
			geometry_msgs::Vector3 dimension;
			getClusterBoundingBox3D(pc2, poseStamped, dimension);

			ostringstream id;
			id << "object " << i;
			collision_object.id = id.str().c_str();

			ROS_INFO_STREAM("Object id: " << collision_object.id);

			collision_object.operation = moveit_msgs::CollisionObject::REMOVE;

			pub_collision_object.publish(collision_object);

			collision_object.operation = moveit_msgs::CollisionObject::ADD;
			collision_object.primitives.resize(1);
			collision_object.primitives[0].type = shape_msgs::SolidPrimitive::BOX;
			collision_object.primitives[0].dimensions.resize(shape_tools::SolidPrimitiveDimCount<shape_msgs::SolidPrimitive::BOX>::value);
			collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_X] = dimension.x;
			collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_Y] = dimension.y;
			collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_Z] = dimension.z;
			collision_object.primitive_poses.resize(1);
			collision_object.primitive_poses[0] = poseStamped.pose;


			pub_collision_object.publish(collision_object);

		}



*/




		//call object pickup
		ROS_INFO("Calling the pickup action");

		manipulation_msgs::Grasp grasp = generateGrasp();

		//group->setSupportSurfaceName("table");

		geometry_msgs::PoseStamped p;

		  p.header.frame_id = KATANA_BASE_LINK;
		  p.pose.position.x = 0.34916;
		  p.pose.position.y = 0;
		  p.pose.position.z = 0.83;
		  /*
		  p.pose.orientation.x = 0.706717;
		  p.pose.orientation.y = 0.000249;
		  p.pose.orientation.z = 0.707496;
		  p.pose.orientation.w = 0.000773;
		  */
		  p.pose.orientation.x = 0;
		  p.pose.orientation.y = 0;
		  p.pose.orientation.z = 0;
		  p.pose.orientation.w = 1;


		  visualization_msgs::Marker marker;
			marker.pose = p.pose;

			//marker.pose = normalPose.pose;
			marker.header.frame_id = KATANA_BASE_LINK;
			marker.id = 7;
			marker.ns = "grasp";
			marker.header.stamp = ros::Time::now();
			marker.action = visualization_msgs::Marker::ADD;
			marker.lifetime = ros::Duration();
			marker.type = visualization_msgs::Marker::ARROW;
			marker.scale.x = 0.05;
			marker.scale.y = 0.005;
			marker.scale.z = 0.005;
			marker.color.r = 0;
			marker.color.g = 1;
			marker.color.b = 0;
			marker.color.a = 1.0;
			vis_marker_publisher.publish(marker);

		//group->setPoseReferenceFrame(KATANA_BASE_LINK);
		//group->setRPYTarget();
		//group->setPositionTarget(p.pose.position.x,p.pose.position.y,p.pose.position.z);
		//group->setPositionTarget(grasp.grasp_pose.pose.position.x,grasp.grasp_pose.pose.position.y,grasp.grasp_pose.pose.position.z);

		//group->setGoalTolerance(0.8);
		//group->setOrientationTarget(grasp.grasp_pose.pose.orientation.x, grasp.grasp_pose.pose.orientation.y ,grasp.grasp_pose.pose.orientation.z ,grasp.grasp_pose.pose.orientation.w);
		//group->setOrientationTarget(p.pose.orientation.x, p.pose.orientation.y ,p.pose.orientation.z ,p.pose.orientation.w);
		ROS_INFO("Set rpy target to 0,0,0");
		group->setRPYTarget(0,0,0);

		ROS_INFO_STREAM("End effector link: " << group->getEndEffectorLink());
		//group->setPoseTarget(p.pose);
		//group->setPoseTarget(grasp.grasp_pose.pose);



		group->move();
		//group->pick("object 4", grasp);
		//group->pick("object 4");

		std::vector<double> rpy = group->getCurrentRPY("katana_motor5_wrist_roll_link");

		ROS_INFO_STREAM("End effector link: " << rpy.at(0)<< rpy.at(1) << rpy.at(2));


		ROS_INFO("Pickup done");

		object_in_gripper = 1;

		move_arm_out_of_the_way();

		return 0;

	}

	void getClusterBoundingBox3D(const sensor_msgs::PointCloud2 &cluster,
							  geometry_msgs::PoseStamped &pose_stamped,
							  geometry_msgs::Vector3 &dimensions)
	{
	  ROS_INFO("GetClusterBoundingBox3D");

	  object_manipulation_msgs::FindClusterBoundingBox2 srv;
	  srv.request.cluster = cluster;
	  if (!cluster_bounding_box2_3d_client_.call(srv.request, srv.response))
	  {
	    ROS_ERROR("Failed to call cluster bounding box client");
	    throw CollisionMapException("Failed to call cluster bounding box client");
	  }
	  pose_stamped = srv.response.pose;
	  dimensions = srv.response.box_dims;
	  if (dimensions.x == 0.0 && dimensions.y == 0.0 && dimensions.z == 0.0)
	  {
	    ROS_ERROR("Cluster bounding box 2 3d client returned an error (0.0 bounding box)");
	    throw CollisionMapException("Bounding box computation failed");
	  }
	}

	manipulation_msgs::Grasp generateGrasp() {

		/*


		sensor_msgs::JointState pre_grasp_joint_state_;
		sensor_msgs::JointState grasp_joint_state_;

		pre_grasp_joint_state_.name.push_back("katana_l_finger_joint");
		pre_grasp_joint_state_.name.push_back("katana_r_finger_joint");
		pre_grasp_joint_state_.position.push_back(0.30);
		pre_grasp_joint_state_.position.push_back(0.30);
		pre_grasp_joint_state_.effort.push_back(100.0);
		pre_grasp_joint_state_.effort.push_back(100.0);

		grasp_joint_state_.name = pre_grasp_joint_state_.name;
		grasp_joint_state_.position.push_back(-0.44);
		grasp_joint_state_.position.push_back(-0.44);
		grasp_joint_state_.effort.push_back(90.0);
		grasp_joint_state_.effort.push_back(90.0);

*/
		const double STANDOFF = -0.07;

		tf::Vector3 position;
		position.setX(normalPoseRobotFrame.pose.position.x);
		position.setY(normalPoseRobotFrame.pose.position.y);
		position.setZ(normalPoseRobotFrame.pose.position.z);


		tf::Transform standoff_trans;
		  standoff_trans.setOrigin(tf::Vector3(position.getX(), position.getY() , 0).normalize()*STANDOFF);
		  standoff_trans.setRotation(tf::createIdentityQuaternion());

		//tf::poseStampedMsgToTF(normalPoseRobotFrame, position);


		position = standoff_trans * position;


		geometry_msgs::PoseStamped p;
		p.header.frame_id = KATANA_BASE_LINK;
		p.pose.position.x = position.getX();
		p.pose.position.y = position.getY();
		p.pose.position.z = position.getZ();



		//#make the normal pose graspable by the Katana 5DOF gripper (Yaw missing)

		// Convert quaternion to RPY.
		tf::Quaternion q;
		double roll, pitch, yaw;
		tf::quaternionMsgToTF(normalPoseRobotFrame.pose.orientation, q);

		tf::Matrix3x3(q).getRPY(roll, pitch, yaw);


		//determine yaw which is compatible with the Katana 300 180 kinematics.
		yaw = atan2(position.getY(),
				position.getX());

		tf::Quaternion quat = tf::createQuaternionFromRPY(0, pitch, yaw);

		p.pose.orientation.x = quat.getX();
		p.pose.orientation.y = quat.getY();
		p.pose.orientation.z = quat.getZ();
		p.pose.orientation.w = quat.getW();

		manipulation_msgs::Grasp g;
		g.grasp_pose = p;

		g.approach.direction.vector.x = 1.0;
		g.approach.direction.header.frame_id = WRIST_JOINT;
		g.approach.min_distance = 0.2;
		g.approach.desired_distance = 0.4;

		g.retreat.direction.header.frame_id = BASE_FOOTPRINT;
		g.retreat.direction.vector.z = 1.0;
		g.retreat.min_distance = 0.1;
		g.retreat.desired_distance = 0.25;

		g.pre_grasp_posture.name.resize(1, FINGER_JOINT);
		g.pre_grasp_posture.position.resize(1);
		g.pre_grasp_posture.position[0] = 1;

		g.grasp_posture.name.resize(1, FINGER_JOINT);
		g.grasp_posture.position.resize(1);
		g.grasp_posture.position[0] = 0;


		ROS_DEBUG_STREAM("Grasp frame id: " << g.grasp_pose.header.frame_id);

		visualization_msgs::Marker marker;
		marker.pose = g.grasp_pose.pose;

		//marker.pose = normalPose.pose;
		marker.header.frame_id = KATANA_BASE_LINK;
		marker.id = 7;
		marker.ns = "grasp";
		marker.header.stamp = ros::Time::now();
		marker.action = visualization_msgs::Marker::ADD;
		marker.lifetime = ros::Duration();
		marker.type = visualization_msgs::Marker::ARROW;
		marker.scale.x = 0.05;
		marker.scale.y = 0.005;
		marker.scale.z = 0.005;
		marker.color.r = 0;
		marker.color.g = 1;
		marker.color.b = 0;
		marker.color.a = 1.0;
		vis_marker_publisher.publish(marker);

		return g;
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
		if (place()) {
			return false;
		}
		return true;
	}

	int place() {

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
		return 0;
	}

	void set_joint_goal() {


		group->setJointValueTarget("katana_motor1_pan_joint", -1.51);
		group->setJointValueTarget("katana_motor2_lift_joint", 2.13549384276445);
		group->setJointValueTarget("katana_motor3_lift_joint", -2.1556486321117725);
		group->setJointValueTarget("katana_motor4_lift_joint", -1.971949347057968);
		group->setJointValueTarget("katana_motor5_wrist_roll_joint", 0.0);

		return;
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
		group->setNamedTarget("home rotated");


		//TODO: move doesn't return??
		group->asyncMove();

		//clear_collision_map();

		//Todo: check if worked

		ROS_INFO("Arm moved out of the way.");

		return 1;
	}

	int clear_collision_map() {

		//if (sim)
		ros::Duration(2.0).sleep(); // only necessary for Gazebo (the simulated Kinect point cloud lags, so we need to wait for it to settle)

		// ----- reset collision map
		ROS_INFO("Clearing collision map");
		std_srvs::Empty empty;
		if (!collider_reset_srv.call(empty)) {
			ROS_ERROR("Collider reset service failed");
			return -1;
		}
		//if (sim)
			ros::Duration(3.0).sleep(); // wait for collision map to be completely cleared

		return 0;

	}



	/**
	 * return nearest object to point
	 */
	void receive_clicked_point_CB(
			const geometry_msgs::PointStamped::ConstPtr& msg) {
		ROS_INFO(
				"Point received: x: %f, y: %f, z: %f ", msg->point.x, msg->point.y, msg->point.z);

		//Throw away old received clicked points
		if ((ros::Time::now().sec - msg->header.stamp.sec)
				> message_receive_dead_time_in_sec) {
			return;
		}

		set_pickup_point(msg->point);

		clicked = true;

		visualization_msgs::Marker marker;

		/*
		marker.pose.position.x = msg->point.x;
		marker.pose.position.y = msg->point.y;
		marker.pose.position.z = msg->point.z;

		marker.header.frame_id = msg->header.frame_id;
		marker.id = 0;
		marker.ns = "selection";
		marker.header.stamp = ros::Time::now();
		marker.action = visualization_msgs::Marker::ADD;
		marker.lifetime = ros::Duration();
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.scale.x = 0.01;
		marker.scale.y = 0.01;
		marker.scale.z = 0.01;
		marker.color.r = 1;
		marker.color.g = 0;
		marker.color.b = 0;
		marker.color.a = 1.0;
		vis_marker_publisher.publish(marker);
		*/

	}

	void receive_cloud_CB(const sensor_msgs::PointCloud2ConstPtr& input_cloud) {

		if (clicked) {

			ROS_INFO("Receive cloud Cb clicked");

			visualization_msgs::Marker marker;


			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

			//pcl::PointCloud<pcl::PointXYZ> *cloud = new pcl::PointCloud<pcl::PointXYZ>();



			pcl::fromROSMsg(*input_cloud, *cloud);



			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud(cloud);

			//transform selected point from robot frame (BASE_FOOTPRINT) to Kinect frame (/kinect_rgb_optical_frame)
			tf::Vector3 searchPointInRobotFrame;

			tf::pointMsgToTF(desired_pickup_point, searchPointInRobotFrame);

			tf::StampedTransform transformRobotToPointCloud;

			try {
				tf_listener->lookupTransform(input_cloud->header.frame_id,
						BASE_FOOTPRINT, ros::Time(0),
						transformRobotToPointCloud);
			} catch (tf::TransformException ex) {
				ROS_ERROR("%s", ex.what());
			}

			tf::Vector3 searchPointPointCloudFrame = transformRobotToPointCloud
					* searchPointInRobotFrame;

			pcl::PointXYZ searchPoint;

			searchPoint.x = searchPointPointCloudFrame.getX();
			searchPoint.y = searchPointPointCloudFrame.getY();
			searchPoint.z = searchPointPointCloudFrame.getZ();

			float radius = 0.005;

			ROS_INFO(
					"Search searchPointWorldFrame set to: x: %f, y: %f, z: %f ", searchPoint.x, searchPoint.y, searchPoint.z);

			// Neighbors within radius search

			std::vector<int> pointIdxRadiusSearch;
			std::vector<float> pointRadiusSquaredDistance;

			ROS_DEBUG_STREAM(
					"Input cloud frame id: " << input_cloud->header.frame_id);

			if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
					pointRadiusSquaredDistance) > 0) {
				for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
					//ROS_DEBUG_STREAM(
					//		"   " << cloud->points[pointIdxRadiusSearch[i]].x << " " << cloud->points[pointIdxRadiusSearch[i]].y << " " << cloud->points[pointIdxRadiusSearch[i]].z << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl);

					marker.pose.position.x =
							cloud->points[pointIdxRadiusSearch[i]].x;
					marker.pose.position.y =
							cloud->points[pointIdxRadiusSearch[i]].y;
					marker.pose.position.z =
							cloud->points[pointIdxRadiusSearch[i]].z;

					//show markers in kinect frame
					marker.header.frame_id = input_cloud->header.frame_id;
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

				normalEstimator.computePointNormal(*cloud, pointIdxRadiusSearch,
						plane_parameters, curvature);

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
						input_cloud->header.frame_id;

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


				ROS_DEBUG_STREAM("Pose in Point Cloud Frame: " << normalPosePointCloudFrame.pose);

				//transform normal pose to Katana base

				try{
					//todo: set back to katana base link for arc tan calc?
				tf_listener->transformPose(KATANA_BASE_LINK,
						normalPosePointCloudFrame, normalPoseRobotFrame);
				}catch(const tf::TransformException &ex){

					ROS_ERROR("%s", ex.what());

				}catch(const std::exception &ex) {

					ROS_ERROR("%s", ex.what());

				}

				ROS_DEBUG_STREAM("Pose in Katana base frame: " << normalPoseRobotFrame.pose);
				ROS_DEBUG_STREAM("Katana base frame frame id: " << normalPoseRobotFrame.header.frame_id);

				marker.pose = normalPoseRobotFrame.pose;

				//marker.pose = normalPose.pose;
				marker.header.frame_id = KATANA_BASE_LINK;
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

				ROS_DEBUG_STREAM("Normal marker published ");

			}

			clicked = false;


			if(object_in_gripper){
				place();
			}else{
				pickup();
			}

		}
	}


	/*
	bool nearest_object(
			std::vector<object_manipulation_msgs::GraspableObject>& objects,
			geometry_msgs::PointStamped& reference_point, int& object_ind) {


		geometry_msgs::PointStamped point;

		// convert point to base_link frame
		tf_listener->transformPoint("/base_link", reference_point, point);

		ROS_INFO_STREAM(
				"Pickup Point test which object is nearest: " << point.point.x << " " << point.point.y << " " << point.point.z);

		// find the closest object
		double nearest_dist = 1e6;
		int nearest_object_ind = -1;

		for (size_t i = 0; i < objects.size(); ++i) {
			sensor_msgs::PointCloud cloud;
			tf_listener->transformPointCloud("/base_link", objects[i].cluster,
					cloud);

			// calculate average
			float x = 0.0, y = 0.0, z = 0.0;
			for (size_t j = 0; j < cloud.points.size(); ++j) {
				x += cloud.points[j].x;
				y += cloud.points[j].y;
				z += cloud.points[j].z;
			}
			x /= cloud.points.size();
			y /= cloud.points.size();
			z /= cloud.points.size();

			double dist = sqrt(
					pow(x - point.point.x, 2.0) + pow(y - point.point.y, 2.0)
							+ pow(z - point.point.z, 2.0));
			if (dist < nearest_dist) {
				nearest_dist = dist;
				nearest_object_ind = i;
			}
		}

		if (nearest_object_ind > -1) {
			ROS_INFO(
					"nearest object ind: %d (distance: %f)", nearest_object_ind, nearest_dist);
			object_ind = nearest_object_ind;
			return true;
		} else {
			ROS_ERROR("No nearby objects. Unable to select grasp target");
			return false;
		}


	}
	*/

};

int main(int argc, char **argv) {
	//initialize the ROS node
	ros::init(argc, argv, "pick_and_place_app");

	ros::NodeHandle nh;

	Pick_and_place_app *app = new Pick_and_place_app(&nh);

	ROS_INFO("Subscripe point cloud and clicked_point");

	// Create a ROS subscriber for the input point cloud
	ros::Subscriber point_cloud_subscriber = nh.subscribe(
			"/kinect/depth_registered/points", 1,
			&Pick_and_place_app::receive_cloud_CB, app);

	ros::Subscriber rviz_click_subscriber = nh.subscribe(
			"/clicked_point", 1000, &Pick_and_place_app::receive_clicked_point_CB, app);


	ros::spin();

	return 0;
}

