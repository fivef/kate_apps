#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <dynamic_reconfigure/DoubleParameter.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>
#include <tf/transform_listener.h>



void receive_imu_message_CB(const sensor_msgs::ImuConstPtr& imu_msg){

		// Convert quaternion to RPY.
		tf::Quaternion q;
		double roll, pitch, yaw;
		tf::quaternionMsgToTF(imu_msg->orientation, q);
		tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

		ROS_INFO_STREAM("Orientation: " << roll << " " << pitch << " " << yaw);

		dynamic_reconfigure::ReconfigureRequest srv_req;
		dynamic_reconfigure::ReconfigureResponse srv_resp;
		dynamic_reconfigure::DoubleParameter double_param;
		dynamic_reconfigure::Config conf;

		double_param.name = "kurtana_pitch_joint";
		double_param.value = pitch;
		conf.doubles.push_back(double_param);

		double_param.name = "kurtana_roll_joint";
		double_param.value = yaw;
		conf.doubles.push_back(double_param);

		srv_req.config = conf;

		ros::service::call("/joint_commander/set_parameters", srv_req, srv_resp);

	}


int main(int argc, char **argv) {
	//initialize the ROS node
	ros::init(argc, argv, "android_drivers_subscriber");

	ros::NodeHandle nh;

	ros::Subscriber android_sensors_subscriber = nh.subscribe("/android/imu", 1, &receive_imu_message_CB);

	ros::spin();

	return 0;
}
