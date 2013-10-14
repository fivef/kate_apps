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

const std::string PLAN_POINT_CLUSTER_GRASP_SERVICE_NAME =
		"/plan_point_cluster_grasp";

const std::string EVALUATE_POINT_CLUSTER_GRASP_SERVICE_NAME =
		"/evaluate_point_cluster_grasps";

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
	static const int DEBUG = true;

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

	geometry_msgs::PoseStamped currentMarkerPose;

	std::vector<manipulation_msgs::Grasp> pickup_grasps;

	std::string object_to_manipulate;
	int object_to_manipulate_index;

	geometry_msgs::Point object_to_manipulate_position;

	move_group_interface::MoveGroup *group;

	ros::Publisher pub_collision_object;

	std::vector<geometry_msgs::PointStamped> object_positions;

	std::vector<sensor_msgs::PointCloud> segmented_clusters;

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

	actionlib::SimpleActionClient<manipulation_msgs::GraspPlanningAction> plan_point_cluster_grasp_action_client;

	ros::NodeHandle nh;

	boost::shared_ptr<interactive_markers::InteractiveMarkerServer> server;
	interactive_markers::MenuHandler menu_handler;

public:

	int object_in_gripper;
	//only calculate normal once after click
	int clicked;

	Pick_and_place_app(ros::NodeHandle *_nh) :
			plan_point_cluster_grasp_action_client(
					PLAN_POINT_CLUSTER_GRASP_SERVICE_NAME, true)

	{

		nh = *_nh;

		//get parameters from parameter server
		nh.param<int>("sim", sim, 0);

		nh.param<int>("DIRECTLY_SELECT_GRASP_POSE", DIRECTLY_SELECT_GRASP_POSE,
				1);

		//the distance between the surface of the object to grasp and the GRIPPER_FRAME origin
		nh.param<double>("OBJECT_GRIPPER_STANDOFF", STANDOFF, -0.1);

		nh.param<std::string>("ARM_BASE_LINK", ARM_BASE_LINK, "jaco_base_link");

		nh.param<std::string>("BASE_LINK", BASE_LINK, "/base_link");

		nh.param<std::string>("GRIPPER_FRAME", GRIPPER_FRAME,
				"jaco_gripper_tool_frame");
		nh.param<std::string>("FINGER_JOINT", FINGER_JOINT,
				"jaco_finger_joint_1");
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

		//group->setPoseReferenceFrame(BASE_LINK);

		//group->setPlanningTime(5.0);

		//group->setGoalTolerance(0.01);
		//group->setGoalOrientationTolerance(0.005);

		group->allowReplanning(true);

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

		cluster_bounding_box2_3d_client_ = nh.serviceClient<
				object_manipulation_msgs::FindClusterBoundingBox2>(
				CLUSTER_BOUNDING_BOX2_3D_NAME, true);

		evaluate_point_cluster_grasp_srv_client = nh.serviceClient<
				manipulation_msgs::GraspPlanning>(
				EVALUATE_POINT_CLUSTER_GRASP_SERVICE_NAME, true);

		vis_marker_publisher = nh.advertise<visualization_msgs::Marker>(
				"pick_and_place_markers", 128);

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

		ROS_INFO("Get nearest object");

		detect_objects_on_table();

		find_nearest_object();

		generate_grasps_for_nearest_cluster();

		ROS_INFO_STREAM(
				"Picking up Object: " << object_to_manipulate << " with " << pickup_grasps.size() << " grasps to try");

		setMarkerPoseToFirstGrasp();

		pickup();

		object_in_gripper = 1;

		return 0;
	}

	int pickup_with_current_marker_pose() {

		pickup_grasps.resize(1);
		pickup_grasps[0] = generateGraspFromPoseStamped(currentMarkerPose);

		create_dummy_collision_object();

		pickup();

		return 0;
	}

	int move_to_current_marker_pose() {

		pickup_grasps.resize(1);
		pickup_grasps[0] = generateGraspFromPoseStamped(currentMarkerPose);

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

		create_dummy_collision_object();

		pickup();

		return 0;
	}

	int pickup() {

		//group->setSupportSurfaceName("table");

		draw_grasps_to_try();

		group->pick(object_to_manipulate, pickup_grasps);

		object_in_gripper = 1;

		//move_arm_out_of_the_way();

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

	void draw_selected_cluster() {

		std::vector<geometry_msgs::Point32> points =
				segmented_clusters[object_to_manipulate_index].points;

		//draw cluster
		visualization_msgs::Marker marker;

		ROS_INFO_STREAM("Number of points in cluster: " << points.size());

		for (size_t i = 0; i < points.size(); i++) {
			marker.pose.position.x = points[i].x;
			marker.pose.position.y = points[i].y;
			marker.pose.position.z = points[i].z;

			//show markers in kinect frame
			marker.header.frame_id =
					segmented_clusters[object_to_manipulate_index].header.frame_id;
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

	}

	int generate_grasps_for_nearest_cluster() {

		ROS_INFO("Generateing grasps for nearest cluster");
		//draw_selected_cluster();

		//Get grasps

		manipulation_msgs::GraspPlanningGoal graspPlanningGoal;

		graspPlanningGoal.target.reference_frame_id = BASE_LINK;
		graspPlanningGoal.target.cluster =
				segmented_clusters[object_to_manipulate_index];

		graspPlanningGoal.arm_name = "arm";

		plan_point_cluster_grasp_action_client.sendGoal(graspPlanningGoal);

		//wait for the action to return
		bool finished_before_timeout =
				plan_point_cluster_grasp_action_client.waitForResult(
						ros::Duration(30.0));

		if (finished_before_timeout) {
			actionlib::SimpleClientGoalState state =
					plan_point_cluster_grasp_action_client.getState();
			ROS_INFO("Action finished: %s", state.toString().c_str());

			pickup_grasps =
					plan_point_cluster_grasp_action_client.getResult()->grasps;

		} else
			ROS_INFO("Action did not finish before the time out.");

		//exit
		return 0;
	}

	int detect_objects_on_table() {
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

		//save clusters for later

		segmented_clusters = detection_call.response.detection.clusters;

		ROS_INFO_STREAM(
				"Number of clusters found: " << segmented_clusters.size());

		//Remove the table because there are convex hull problems if adding the table to envirnonment
		//detection_call.response.detection.table.convex_hull = shape_msgs::Mesh();

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

		if (process_call.response.collision_object_names.empty()) {
			ROS_ERROR("Tabletop Collision Map Processing error");
			return -1;
		}

		return 0;

	}

	/*
	 * creates a dummy collision object for the pick() function
	 * Takes the pickup_grasp to set the objects pose.
	 */
	void create_dummy_collision_object() {
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
		collision_object.primitive_poses[0] = pickup_grasps[0].grasp_pose.pose;

		pub_collision_object.publish(collision_object);

		//set the current object to manipulate to the generated dummy object
		object_to_manipulate = "dummy";

	}

	/*
	 * Generate a grasp out of a pose
	 */

	manipulation_msgs::Grasp generateGraspFromPoseStamped(
			geometry_msgs::PoseStamped pose) {

		manipulation_msgs::Grasp g;
		g.grasp_pose = pose;

		g.approach.direction.vector.x = 1.0;
		g.approach.direction.header.frame_id = GRIPPER_FRAME;
		g.approach.min_distance = 0.02;
		g.approach.desired_distance = 0.03;

		g.retreat.direction.header.frame_id = BASE_LINK;
		g.retreat.direction.vector.z = 1.0;
		g.retreat.min_distance = 0.02;
		g.retreat.desired_distance = 0.03;

		g.pre_grasp_posture.header.frame_id = BASE_LINK;
		g.pre_grasp_posture.header.stamp = ros::Time::now();
		g.pre_grasp_posture.name.resize(2);
		g.pre_grasp_posture.name[0] = FINGER_JOINT;
		g.pre_grasp_posture.position.resize(2);
		g.pre_grasp_posture.position[0] = -0.5;

		//TODO create params for this
		g.pre_grasp_posture.name[1] = "jaco_finger_joint_2";
		g.pre_grasp_posture.position[1] = -0.5;

		g.grasp_posture.header.frame_id = BASE_LINK;
		g.grasp_posture.header.stamp = ros::Time::now();
		g.grasp_posture.name.resize(2);
		g.grasp_posture.name[0] = FINGER_JOINT;
		g.grasp_posture.position.resize(2);
		g.grasp_posture.position[0] = 0.2;

		g.grasp_posture.name[1] = "jaco_finger_joint_2";
		g.grasp_posture.position[1] = 0.2;

		g.allowed_touch_objects.resize(1);
		g.allowed_touch_objects[0] = "dummy";

		ROS_DEBUG_STREAM("Grasp frame id: " << g.grasp_pose.header.frame_id);

		ROS_DEBUG_STREAM("Grasp Pose" << g.grasp_pose.pose);



		return g;

	}

	void draw_grasps_to_try() {

		for (size_t i = 0; i < pickup_grasps.size(); ++i) {

			visualization_msgs::Marker marker;
			marker.pose = pickup_grasps[i].grasp_pose.pose;

			//marker.pose = normalPose.pose;
			marker.header.frame_id =
					pickup_grasps[i].grasp_pose.header.frame_id;
			marker.id = i;
			marker.ns = "generated_pickup_grasp";
			marker.header.stamp = ros::Time::now();
			marker.action = visualization_msgs::Marker::ADD;
			marker.lifetime = ros::Duration();

			marker.type = Marker::CUBE;
					marker.scale.x = 0.03;
					marker.scale.y = 0.1;
					marker.scale.z = 0.01;
					marker.color.r = 0.5;
					marker.color.g = 0.5;
					marker.color.b = 0.5;
					marker.color.a = 1.0;
			/*
			marker.type = visualization_msgs::Marker::ARROW;
			marker.scale.x = 0.05;
			marker.scale.y = 0.005;
			marker.scale.z = 0.005;
			marker.color.r = 0;
			marker.color.g = 1;
			marker.color.b = 0;
			marker.color.a = 1.0;
			*/
			vis_marker_publisher.publish(marker);
		}

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

		return generateGraspFromPoseStamped(p);
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

			object_to_manipulate = id.c_str();

			return true;

		} else {
			ROS_ERROR("No nearby objects. Unable to select a pickup target");
			return false;
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
		marker.color.a = 1.0;

		return marker;
	}

	InteractiveMarkerControl& makeBoxControl(InteractiveMarker &msg) {
		InteractiveMarkerControl control;
		control.always_visible = true;
		control.markers.push_back(makeBox(msg));
		msg.controls.push_back(control);

		return msg.controls.back();
	}

	void make6DofMarker() {

		//menu
		menu_handler.insert("Pickup Nearest Segmented Object",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("Pickup by Surface Normal",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("Pickup by Marker Pose",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("move_to_current_marker_pose",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));
		menu_handler.insert("reset_collision_environment",
				boost::bind(&Pick_and_place_app::processFeedback, this, _1));

		InteractiveMarker int_marker;
		int_marker.header.frame_id = BASE_LINK;

		tf::Vector3 position;

		position = tf::Vector3(0, 0, 0);
		tf::pointTFToMsg(position, int_marker.pose.position);
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
	}

	void setMarkerPoseToFirstGrasp() {

		InteractiveMarker int_marker;
		server->get("simple_6dof", int_marker);

		int_marker.pose = pickup_grasps[0].grasp_pose.pose;

		currentMarkerPose = pickup_grasps[0].grasp_pose;

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

			//"Pickup Nearest Segmented Object"
			case 1:
				pickup_nearest_segmented_object();
				break;

				//"Pickup by Surface Normal"
			case 2:
				pickup_with_normal();
				break;

				//"Pickup by Marker Pose"
			case 3:
				pickup_with_current_marker_pose();
				break;
				//Move to the current marker pose
			case 4:
				move_to_current_marker_pose();
				break;

			case 5:

				detect_objects_on_table();
				break;
			}
			break;

		case visualization_msgs::InteractiveMarkerFeedback::POSE_UPDATE:
			ROS_INFO_STREAM(
					s.str() << ": pose changed" << "\nposition = " << feedback->pose.position.x << ", " << feedback->pose.position.y << ", " << feedback->pose.position.z << "\norientation = " << feedback->pose.orientation.w << ", " << feedback->pose.orientation.x << ", " << feedback->pose.orientation.y << ", " << feedback->pose.orientation.z << "\nframe: " << feedback->header.frame_id << " time: " << feedback->header.stamp.sec << "sec, " << feedback->header.stamp.nsec << " nsec");

			currentMarkerPose.pose.orientation = feedback->pose.orientation;
			currentMarkerPose.pose.position = feedback->pose.position;
			currentMarkerPose.header.frame_id = feedback->header.frame_id;

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

	ROS_INFO("Pick and Place v 0.2 ready to take commands.");

	//Async Spinner for Clicks_queue, because the moveit functions like pickup, move don't return in synchronous callbacks
	ros::AsyncSpinner spinner(0, &clicks_queue);
	spinner.start();

	ros::spin();

	return 0;
}

