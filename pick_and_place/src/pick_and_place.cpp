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

#include <ros/callback_queue.h>

#include <object_manipulation_msgs/FindClusterBoundingBox2.h>
#include <tabletop_object_detector/TabletopDetection.h>
#include <tabletop_collision_map_processing/TabletopCollisionMapProcessing.h>

//for PTU dynamic reconfigure
#include <dynamic_reconfigure/DoubleParameter.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>

// MoveIt!
#include <moveit/pick_place/pick_place.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/move_group_interface/move_group.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <moveit_msgs/PlanningSceneComponents.h>
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

const std::string GET_PLANNING_SCENE_SERVICE_NAME = "/get_planning_scene";


static const std::string CLUSTER_BOUNDING_BOX2_3D_NAME =
		"find_cluster_bounding_box2_3d";

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
	static const int message_receive_dead_time_in_sec = 5;

	//1 normal directly selects grasp pose / 0 nearest objects to selected point is grasped
	int DIRECTLY_SELECT_GRASP_POSE;

	static const float place_position_tolerance_in_meter = 0.03;
	static const float place_planner_step_size_in_meter = 0.005;

	//the distance in y direction from the position where the object was picked up to the place position
	static const float place_offset = 0.20;

	static const size_t NUM_JOINTS = 5;

	double STANDOFF;

	std::string ARM_BASE_LINK;

	std::string BASE_LINK;

	std::string GRIPPER_FRAME;

	std::string FINGER_JOINT;

	std::string ARM_NAME;

	static const int NORMAL = 1;
	static const int SEGMENTED_OBJECT_SELECTION = 2;
	std::string WORKING_MODE;

	geometry_msgs::PointStamped desired_pickup_point;

	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input_point_cloud;

	int running;

	ros::Publisher vis_marker_publisher;

	tf::TransformListener *tf_listener;

	geometry_msgs::PoseStamped normalPoseRobotFrame;

	pcl::PointXYZ normal;

	manipulation_msgs::Grasp generated_pickup_grasp;

	std::string object_to_manipulate;
	geometry_msgs::Point object_to_manipulate_position;


	move_group_interface::MoveGroup *group;



	ros::Publisher pub_collision_object;

	std::vector<geometry_msgs::PointStamped> object_positions;

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

	ros::NodeHandle nh;

public:

	int object_in_gripper;
	//only calculate normal once after click
	int clicked;

	Pick_and_place_app(ros::NodeHandle *_nh) {

		nh = *_nh;

		//get parameters from parameter server
		nh.param<int>("sim", sim, 0);

		nh.param<int>("DIRECTLY_SELECT_GRASP_POSE", DIRECTLY_SELECT_GRASP_POSE,
				1);

		//the distance between the surface of the object to grasp and the GRIPPER_FRAME origin
		nh.param<double>("OBJECT_GRIPPER_STANDOFF", STANDOFF, 0.08);

		nh.param<std::string>("ARM_BASE_LINK", ARM_BASE_LINK,
				"katana_base_link");

		nh.param<std::string>("BASE_LINK", BASE_LINK, "base_link");

		nh.param<std::string>("GRIPPER_FRAME", GRIPPER_FRAME,
				"katana_gripper_tool_frame");
		nh.param<std::string>("FINGER_JOINT", FINGER_JOINT,
				"katana_l_finger_joint");
		nh.param<std::string>("ARM_NAME", ARM_NAME, "arm");

		clicked = 0;

		object_in_gripper = 0;

		pcl_input_point_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

		// create TF listener
		tf_listener = new tf::TransformListener();

		pub_collision_object = nh.advertise<moveit_msgs::CollisionObject>(
				"collision_object", 10);

		ros::WallDuration(1.0).sleep();

		group = new move_group_interface::MoveGroup(ARM_NAME);

		//group->setPoseReferenceFrame(BASE_LINK);

		//group->setPlanningTime(5.0);

		group->setGoalTolerance(0.01);
		group->setGoalOrientationTolerance(0.005);

		group->allowReplanning(true);

		group->setWorkspace(-0.5,-0.6,-0.3,0.6,0.6,1.5);

		//wait for get planning scene server

		while(!ros::service::waitForService (GET_PLANNING_SCENE_SERVICE_NAME, ros::Duration(2.0)) && nh.ok()){
			ROS_INFO("Waiting for get planning scene service to come up");
		}
		if (!nh.ok())
			exit(0);
		get_planning_scene_srv = nh.serviceClient<moveit_msgs::GetPlanningScene>(GET_PLANNING_SCENE_SERVICE_NAME, true);


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


		cluster_bounding_box2_3d_client_ = nh.serviceClient<
				object_manipulation_msgs::FindClusterBoundingBox2>(
				CLUSTER_BOUNDING_BOX2_3D_NAME, true);


		vis_marker_publisher = nh.advertise<visualization_msgs::Marker>(
					"pick_and_place_markers", 128);


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

	int pickup() {
		/*
		 // ----- pick up object near point (in meter) relative to base_footprint
		 geometry_msgs::PointStamped pickup_point;
		 pickup_point.header.frame_id = KURTANA_BASE_LINK;
		 pickup_point.point.x = desired_pickup_point.x;
		 pickup_point.point.y = desired_pickup_point.y;
		 pickup_point.point.z = desired_pickup_point.z;
		 */

		// ----- call the tabletop detection
		ROS_INFO("Calling tabletop detector");
		tabletop_object_detector::TabletopDetection detection_call;
		//we want recognized database objects returned
		//set this to false if you are using the pipeline without the database
		detection_call.request.return_models = false;

		//we want the individual object point clouds returned as well
		detection_call.request.return_clusters = false;

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

		tabletop_collision_map_processing::TabletopCollisionMapProcessing process_call;

		process_call.request.detection_result = detection_call.response.detection;
		process_call.request.reset_collision_models = false;
		process_call.request.reset_attached_models = false;

		if (!collision_processing_srv.call(process_call)) {
				ROS_ERROR("Tabletop Collision Map Processing failed");
				return -1;
		}

		ROS_INFO_STREAM("Found objects count: " << process_call.response.collision_object_names.size());

		if (process_call.response.collision_object_names.empty()) {
				ROS_ERROR("Tabletop Collision Map Processing error");
				return -1;
		}

		/*

		ROS_INFO("Add table to planning scene");

		moveit_msgs::CollisionObject collision_object;

		collision_object.header.stamp = ros::Time::now();
		collision_object.header.frame_id =
				detection_call.response.detection.table.pose.header.frame_id;



		ROS_INFO_STREAM("Add clusters");

		object_positions.clear();

		//add objects to planning scene
		for (unsigned int i = 0;
				i < detection_call.response.detection.clusters.size(); i++) {

			sensor_msgs::PointCloud2 pc2;
			sensor_msgs::convertPointCloudToPointCloud2(
					detection_call.response.detection.clusters[i], pc2);
			geometry_msgs::PoseStamped poseStamped;
			geometry_msgs::Vector3 dimension;
			getClusterBoundingBox3D(pc2, poseStamped, dimension);

			geometry_msgs::PointStamped point;

			point.header.frame_id = poseStamped.header.frame_id;
			point.point = poseStamped.pose.position;

			object_positions.push_back(point);

			ostringstream id;
			id << "object " << i;
			collision_object.id = id.str().c_str();

			ROS_INFO_STREAM("Object id: " << collision_object.id);

			collision_object.operation = moveit_msgs::CollisionObject::REMOVE;

			pub_collision_object.publish(collision_object);

			collision_object.operation = moveit_msgs::CollisionObject::ADD;
			collision_object.primitives.resize(1);
			collision_object.primitives[0].type =
					shape_msgs::SolidPrimitive::BOX;
			collision_object.primitives[0].dimensions.resize(
					shape_tools::SolidPrimitiveDimCount<
							shape_msgs::SolidPrimitive::BOX>::value);
			collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_X] =
					dimension.x;
			collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_Y] =
					dimension.y;
			collision_object.primitives[0].dimensions[shape_msgs::SolidPrimitive::BOX_Z] =
					dimension.z;
			collision_object.primitive_poses.resize(1);
			collision_object.primitive_poses[0] = poseStamped.pose;

			pub_collision_object.publish(collision_object);

		}

		*/

		//group->setSupportSurfaceName("table");

		//call object pickup
		ROS_INFO("Calling the pickup action");

		generated_pickup_grasp = generateGrasp();

		ROS_INFO("Get nearest object");

		object_to_manipulate = nearest_object();


		//test

		/*
[DEBUG] [1378456372.242171284]: IK pose: position:
  x: 0.44866
  y: -0.185202
  z: 0.392208
orientation:
  x: 0.00125435
  y: -0.00632681
  z: -0.194471
  w: 0.980887

[DEBUG] [1378456372.242351955]: Found 4 solutions from IKFast
[DEBUG] [1378456372.242443678]: Sol 0: -3.914825e-01   1.548975e+00   -1.572923e+00   -3.587233e-02   -4.921995e-03   2.618548e-322
[DEBUG] [1378456372.243036938]: Solution passes callback


		geometry_msgs::PoseStamped p;
		p.header.frame_id = ARM_BASE_LINK;
		p.pose.position.x = 0.44866;
		p.pose.position.y = -0.185202;
		p.pose.position.z = 0.392208;


		p.pose.orientation.x = 0;
		p.pose.orientation.y = 0;
		p.pose.orientation.z = 0;
		p.pose.orientation.w = 1;

		group->setPoseTarget(p,GRIPPER_FRAME);

		group->move();
*/
		//group->pick(object_to_manipulate);

		ROS_INFO_STREAM("Picking up Object: " << object_to_manipulate);

		group->pick(object_to_manipulate, generated_pickup_grasp);

		ROS_INFO("Pick returned!!!!!11111 OMGWTFIT");

		//group->place(object_to_manipulate);

		std::vector<double> rpy = group->getCurrentRPY(
				group->getEndEffectorLink());

		ROS_INFO_STREAM(
				"End effector link: " << rpy.at(0)<< rpy.at(1) << rpy.at(2));




		object_in_gripper = 1;

		move_arm_out_of_the_way();



		return 0;

	}

	void getClusterBoundingBox3D(const sensor_msgs::PointCloud2 &cluster,
			geometry_msgs::PoseStamped &pose_stamped,
			geometry_msgs::Vector3 &dimensions) {
		ROS_INFO("GetClusterBoundingBox3D");

		object_manipulation_msgs::FindClusterBoundingBox2 srv;
		srv.request.cluster = cluster;
		if (!cluster_bounding_box2_3d_client_.call(srv.request, srv.response)) {
			ROS_ERROR("Failed to call cluster bounding box client");
			throw CollisionMapException(
					"Failed to call cluster bounding box client");
		}
		pose_stamped = srv.response.pose;
		dimensions = srv.response.box_dims;
		if (dimensions.x == 0.0 && dimensions.y == 0.0 && dimensions.z == 0.0) {
			ROS_ERROR(
					"Cluster bounding box 2 3d client returned an error (0.0 bounding box)");
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

		ROS_DEBUG_STREAM("make pose reachable input: " << p.pose);
		make_pose_reachable_by_5DOF_katana(p);
		ROS_DEBUG_STREAM("make pose reachable output: " << p.pose);

		manipulation_msgs::Grasp g;
		g.grasp_pose = p;

		g.approach.direction.vector.x = 1.0;
		g.approach.direction.header.frame_id = GRIPPER_FRAME;
		g.approach.min_distance = 0.05;
		g.approach.desired_distance = 0.07;

		g.retreat.direction.header.frame_id = BASE_LINK;
		g.retreat.direction.vector.z = 1.0;
		g.retreat.min_distance = 0.05;
		g.retreat.desired_distance = 0.07;

		g.pre_grasp_posture.header.frame_id = BASE_LINK;
		g.pre_grasp_posture.header.stamp = ros::Time::now();
		g.pre_grasp_posture.name.resize(1);
		g.pre_grasp_posture.name[0] = FINGER_JOINT;
		g.pre_grasp_posture.position.resize(1);
		g.pre_grasp_posture.position[0] = 0.30;

		g.grasp_posture.header.frame_id = BASE_LINK;
		g.grasp_posture.header.stamp = ros::Time::now();
		g.grasp_posture.name.resize(1);
		g.grasp_posture.name[0] = FINGER_JOINT;
		g.grasp_posture.position.resize(1);
		g.grasp_posture.position[0] = -0.44;

		ROS_DEBUG_STREAM("Grasp frame id: " << g.grasp_pose.header.frame_id);

		ROS_DEBUG_STREAM("Grasp Pose" << g.grasp_pose.pose);

		visualization_msgs::Marker marker;
		marker.pose = g.grasp_pose.pose;

		//marker.pose = normalPose.pose;
		marker.header.frame_id = g.grasp_pose.header.frame_id;
		marker.id = 7;
		marker.ns = "generated_pickup_grasp";
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


	void make_pose_reachable_by_5DOF_katana(geometry_msgs::PoseStamped &pose){

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



		//group->place(object_to_manipulate, generated_pickup_grasp.grasp_pose);


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
		p = generated_pickup_grasp.grasp_pose;


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

	void set_joint_goal() {

		group->setJointValueTarget("katana_motor1_pan_joint", -1.51);
		group->setJointValueTarget("katana_motor2_lift_joint",
				2.13549384276445);
		group->setJointValueTarget("katana_motor3_lift_joint",
				-2.1556486321117725);
		group->setJointValueTarget("katana_motor4_lift_joint",
				-1.971949347057968);
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
		group->setNamedTarget("home_stable");

		group->asyncMove();

		//clear_collision_map();

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

		pickup();

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

	string nearest_object() {

		//desired_pickup_point

		moveit_msgs::GetPlanningScene get_planning_scene_call;

		//get all planning scene objects
		get_planning_scene_call.request.components.components = moveit_msgs::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY;



		if (!get_planning_scene_srv.call(get_planning_scene_call)) {
				ROS_ERROR("Get Planning Scene call failed");
				return "";
		}

		if (get_planning_scene_call.response.scene.world.collision_objects.empty()) {
				ROS_ERROR("Get Planning scene returned nothing");
				return "";
		}


		geometry_msgs::PointStamped point;

		// convert point to base_link frame
		tf_listener->transformPoint("/base_link", desired_pickup_point, point);

		ROS_DEBUG_STREAM(
				"Pickup Point test which object is nearest: " << point.point.x << " " << point.point.y << " " << point.point.z);

		// find the closest object
		double nearest_dist = 1e6;
		int nearest_object_ind = -1;

		int number_of_scene_objects = get_planning_scene_call.response.scene.world.collision_objects.size() - 1;

		ROS_INFO_STREAM("Number of Scene Objects: " << number_of_scene_objects);

		for (int i = 0; i < number_of_scene_objects; i++) {

			geometry_msgs::Point object_position_in_base_link_frame = get_planning_scene_call.response.scene.world.collision_objects[i].primitive_poses[0].position;
			ROS_INFO_STREAM("object " << i << " position: " << object_position_in_base_link_frame);
			/*
			geometry_msgs::PointStamped object_position_in_base_link_frame;
			tf_listener->transformPoint("/base_link", object_positions[i],
					object_position_in_base_link_frame);
			*/
			double dist = sqrt(
					pow(
							object_position_in_base_link_frame.x
									- point.point.x, 2.0)
							+ pow(
									object_position_in_base_link_frame.y
											- point.point.y, 2.0)
							+ pow(
									object_position_in_base_link_frame.z
											- point.point.z, 2.0));
			if (dist < nearest_dist) {
				nearest_dist = dist;
				nearest_object_ind = i;
				object_to_manipulate_position = geometry_msgs::Point(object_position_in_base_link_frame);
			}
		}

		if(nearest_object_ind > -1) {
			ROS_INFO("NEAREST");
			ROS_INFO("nearest object ind: %d (distance: %f", nearest_object_ind, nearest_dist);

			//object_to_manipulate_position = get_planning_scene_call.response.scene.world.collision_objects[nearest_object_ind].primitive_poses[0].position;

			ROS_INFO_STREAM("Object Position: " << object_to_manipulate_position);

			string id = get_planning_scene_call.response.scene.world.collision_objects[nearest_object_ind].id;

			return id.c_str();

		} else {
			ROS_ERROR("No nearby objects. Unable to select a pickup target");
			return "";
		}

	}

};

int main(int argc, char **argv) {
	//initialize the ROS node
	ros::init(argc, argv, "pick_and_place_app");

	ros::NodeHandle nh;

	ros::CallbackQueue clicks_queue;

	Pick_and_place_app *app = new Pick_and_place_app(&nh);

	//Advertise pickup_object Service and use async callback queue.
	ros::AdvertiseServiceOptions advertiseServiceOptions = ros::AdvertiseServiceOptions::create<std_srvs::Empty>("pickup_object",boost::bind(&Pick_and_place_app::pickup_callback, app, _1, _2), ros::VoidPtr(), &clicks_queue);
	ros::ServiceServer pickup_object_srv = nh.advertiseService(advertiseServiceOptions);

	//Advertise place_object Service and use async callback queue.
	advertiseServiceOptions = ros::AdvertiseServiceOptions::create<std_srvs::Empty>("place_object",boost::bind(&Pick_and_place_app::place_callback, app, _1, _2), ros::VoidPtr(), &clicks_queue);
	ros::ServiceServer place_object_srv = nh.advertiseService(advertiseServiceOptions);

	//Advertise "move_arm_out_of_the_way" Service and use async callback queue.
	advertiseServiceOptions = ros::AdvertiseServiceOptions::create<std_srvs::Empty>("move_arm_out_of_the_way",boost::bind(&Pick_and_place_app::move_arm_out_of_the_way_callback, app, _1, _2), ros::VoidPtr(), &clicks_queue);
	ros::ServiceServer move_arm_out_of_the_way_srv = nh.advertiseService(advertiseServiceOptions);

	// Create a ROS subscriber for the input point cloud

	ros::Subscriber point_cloud_subscriber = nh.subscribe(
			"/kinect/depth_registered/points", 1,
			&Pick_and_place_app::receive_cloud_CB, app);

	//Async Queue for Clicks_queue, because the moveit functions like pickup, move don't return in synchronous callbacks
	ros::SubscribeOptions options = ros::SubscribeOptions::create<geometry_msgs::PointStamped>("/clicked_point", 1, boost::bind(&Pick_and_place_app::receive_clicked_point_CB, app, _1)
			, ros::VoidPtr(), &clicks_queue);

	ros::Subscriber rviz_click_subscriber = nh.subscribe(options);

	ROS_INFO("Subscriped to point cloud and clicked_point");

	ROS_INFO("Pick and Place v 0.1 ready to take commands.");


	//Async Spinner for Clicks_queue, because the moveit functions like pickup, move don't return in synchronous callbacks
	ros::AsyncSpinner spinner(0, &clicks_queue);
	spinner.start();


	ros::spin();

	return 0;
}

