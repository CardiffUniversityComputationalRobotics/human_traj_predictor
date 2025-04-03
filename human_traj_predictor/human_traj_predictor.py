import math
from collections import defaultdict, deque
import rclpy
from rclpy.node import Node
import torch
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from human_traj_predictor.ped_pred.DataLoader import DataLoader
from human_traj_predictor.ped_pred.pedestrian_trajectory import traj_prediction
from human_traj_predictor.ped_pred.helper import getCoef, cov_mat_generation
from human_traj_predictor.tensor_generator import (
    get_mask_tensor,
    get_odom_tensor,
    get_social_agents_tensor,
    filter_curr_ob,
)
from tidup_move_base_msgs.msg import (
    AgentStatesPrediction,
    AgentStatePrediction,
    PoseWith2DCovariance,
)


class HumanTrajPredictor(Node):
    def __init__(self):
        super().__init__("human_traj_predictor")

        self.recording_period_ = 0.5
        self.sequence_length_ = 6
        self.pred_len_ = 6
        self.max_human_num_ = 5

        self.history_counter_ = 0

        self.agents_data_ = None
        self.odom_data_ = None

        self.ped_traj_pred_ = traj_prediction()
        self.dataloader_ = DataLoader(phase="test")

        # ! SUBSCRIBERS
        self.agent_states_sub_ = self.create_subscription(
            AgentStates,
            "/pedsim_simulator/simulated_agents",
            self.agent_states_callback,
            1,
        )

        self.odom_sub_ = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            1,
        )

        # ! PUBLISHERS
        self.predictions_pub_ = self.create_publisher(MarkerArray, "agents_pred", 10)

        # ! DATA HISTORY FOR MODEL
        self.agents_history = defaultdict(lambda: deque(maxlen=self.sequence_length_))
        self.odom_history = defaultdict(lambda: deque(maxlen=self.sequence_length_))

        # ! TIMERS
        self.recording_timer = self.create_timer(
            self.recording_period_, self.recording_callback
        )

        self.pred_timer = self.create_timer(0.1, self.predict_traj)

    def predict_traj(self):

        if (
            self.agents_data_
            and self.odom_data_
            and self.history_counter_ > self.sequence_length_
        ):

            # ped and robot tensors
            ped_pos = get_social_agents_tensor(
                self.sequence_length_, self.agents_history
            )
            # ped_pos = torch.cat((ped_pos, torch.zeros(6, 55, 5)), dim=1)
            robot_pos = get_odom_tensor(self.sequence_length_, self.odom_history)

            # mask tensors
            ped_mask = get_mask_tensor(
                self.sequence_length_, len(self.agents_history.keys()), True
            )
            # ped_mask = torch.cat((ped_mask, torch.full((6, 55), False)), dim=1)

            veh_mask = get_mask_tensor(self.sequence_length_ * 2, 1, False)
            veh_mask = torch.cat((veh_mask, torch.full((12, 14), False)), dim=1)

            # empty tensors
            veh_pos = torch.rand(self.sequence_length_ * 2, 15, 5)
            robot_plan_env = torch.zeros(self.sequence_length_, 1, 5)

            ob_ped_pos, ob_ped_mask, col_ind_pres_peds = filter_curr_ob(
                ped_pos, ped_mask
            )
            ob_veh_pos, ob_veh_mask, col_ind_pres_vehs = filter_curr_ob(
                veh_pos, veh_mask
            )

            # print("=====================")
            # print("ped_pos shape: ", ped_pos.shape)
            # print("ped_mask shape: ", ped_mask.shape)
            # print("veh_pos shape: ", veh_pos.shape)
            # print("veh_mask shape: ", veh_mask.shape)
            # print("robot_pos shape: ", robot_pos.shape)
            # print("++++++++++++++++++++")

            pred_pos = torch.zeros((self.pred_len_, self.max_human_num_, 5))
            pred_dist = torch.zeros((self.pred_len_, self.max_human_num_, 5))
            pred_cov = (
                torch.eye(2)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.pred_len_, self.max_human_num_, 1, 1)
            )
            # ones for making the inverse possible for not existing peds

            # print("=====================")
            # print(ob_ped_pos)

            with torch.no_grad():
                # ped_pred: (pred_seq_len, num_peds, 5)
                ped_pred, dist_param = self.ped_traj_pred_.forward(
                    ob_ped_pos.cpu(),
                    ob_ped_mask.cpu(),
                    ob_veh_pos.cpu(),
                    ob_veh_mask.cpu(),
                    robot_pos.cpu(),
                    robot_plan_env.cpu(),
                    self.dataloader_.timestamp,
                )

                mux, muy, sx, sy, corr = getCoef(dist_param.cpu())
                scaled_param_dist = torch.stack((mux, muy, sx, sy, corr), 2)
                cov = cov_mat_generation(scaled_param_dist)

                pred_pos[:, col_ind_pres_peds, :] = ped_pred.cpu()
                pred_dist[:, col_ind_pres_peds, :] = dist_param.cpu()
                pred_cov[:, col_ind_pres_peds, :, :] = cov.cpu()

                # print(pred_pos.shape)
                # print(pred_pos)
                print(pred_cov.shape)
                # print(pred_cov)
            self.publish_marker_array(pred_pos.tolist())
            self.publish_agents_prediction(pred_pos.tolist(), pred_cov.tolist())

    def publish_agents_prediction(self, pred_tensor, cov_tensor):
        agent_states_prediction = AgentStatesPrediction()
        for j in range(self.max_human_num_):
            agent_state_prediction = AgentStatePrediction()
            agent_state_prediction.agent_state = self.agents_data_[j]
            for i in range(self.sequence_length_):
                predicted_pose = PoseWith2DCovariance()
                # predicted_pose.pose =
                # predicted_pose.covariance =
                agent_state_prediction.predicted_poses.append(predicted_pose)
            agent_states_prediction.agent_states_prediction.append(
                agent_state_prediction
            )

    def publish_marker_array(self, pred_tensor):
        marker_array = MarkerArray()

        marker_id = 0

        # Loop over each person and create a Marker for them
        for i in range(self.sequence_length_):
            for j in range(self.max_human_num_):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "positions"
                marker.id = marker_id  # Unique ID for each marker (one for each person)
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = pred_tensor[i][j][0]  # position x
                marker.pose.position.y = pred_tensor[i][j][1]  # position y
                marker.pose.position.z = 0.0  # Flat on the ground

                marker.scale.x = 0.2  # Size of the sphere
                marker.scale.y = 0.2
                marker.scale.z = 0.2

                marker.color.a = 1.0  # Fully opaque
                marker.color.r = 1.0  # Red color
                marker.color.g = 0.0
                marker.color.b = 0.0

                # Add the marker to the MarkerArray
                marker_array.markers.append(marker)
                marker_id += 1

            # Publish the MarkerArray
            self.predictions_pub_.publish(marker_array)

    def recording_callback(self):
        # AGENTS RECORDING
        if self.agents_data_:
            for agent in self.agents_data_:
                agent_id = agent.id
                agent_data = (
                    agent.pose.position.x,
                    agent.pose.position.y,
                    agent.twist.linear.x,
                    agent.twist.linear.y,
                    agent.header.stamp.sec + agent.header.stamp.nanosec * 1e-9,
                )
                self.agents_history[agent_id].append(agent_data)

        # ODOM RECORDING
        if self.odom_data_:
            yaw = euler_from_quaternion(
                [
                    self.odom_data_.pose.pose.orientation.x,
                    self.odom_data_.pose.pose.orientation.y,
                    self.odom_data_.pose.pose.orientation.z,
                    self.odom_data_.pose.pose.orientation.w,
                ]
            )[2]

            odom_data = (
                self.odom_data_.pose.pose.position.x,
                self.odom_data_.pose.pose.position.y,
                self.odom_data_.twist.twist.linear.x * math.cos(yaw),
                self.odom_data_.twist.twist.linear.x * math.sin(yaw),
                self.odom_data_.header.stamp.sec
                + self.odom_data_.header.stamp.nanosec * 1e-9,
            )
            self.odom_history[0].append(odom_data)

        self.history_counter_ += 1

    def agent_states_callback(self, msg: AgentStates):
        self.max_human_num_ = len(msg.agent_states)
        self.agents_data_ = msg.agent_states

    def odom_callback(self, msg: Odometry):
        self.odom_data_ = msg


def main(args=None):
    rclpy.init(args=args)
    node = HumanTrajPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
