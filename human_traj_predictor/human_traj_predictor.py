import rclpy
from rclpy.node import Node
import torch
import math
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from collections import defaultdict, deque
from tf_transformations import euler_from_quaternion
from human_traj_predictor.ped_pred.DataLoader import DataLoader
from human_traj_predictor.ped_pred.pedestrian_trajectory import traj_prediction
from human_traj_predictor.ped_pred.helper import getCoef, cov_mat_generation
from human_traj_predictor.tensor_generator import (
    get_mask_tensor,
    get_odom_tensor,
    get_social_agents_tensor,
    filter_curr_ob,
)


class HumanTrajPredictor(Node):
    def __init__(self):
        super().__init__("human_traj_predictor")

        self.recording_period = 0.5
        self.sequence_length = 5
        self.pred_len = 5
        self.max_human_num = 60

        self.agents_data = None
        self.odom_data = None

        print("about to run")

        self.ped_traj_pred = traj_prediction()
        self.dataloader = DataLoader(phase="test")

        print("passed loader")

        self.agent_states_sub = self.create_subscription(
            AgentStates,
            "/pedsim_simulator/simulated_agents",
            self.agent_states_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.agents_history = defaultdict(lambda: deque(maxlen=self.sequence_length))
        self.odom_history = defaultdict(lambda: deque(maxlen=self.sequence_length))

        self.recording_timer = self.create_timer(
            self.recording_period, self.recording_callback
        )

        # self.social_tensor_timer = self.create_timer(0.5, self.get_social_agents_tensor)
        # self.odom_tensor_timer = self.create_timer(0.5, self.get_odom_tensor)
        # self.mask_tensor_timer = self.create_timer(0.5, self.get_mask_tensor)
        self.pred_timer = self.create_timer(0.5, self.predict_traj)
        print("ready to predict")

    def predict_traj(self):
        if self.agents_data and self.odom_data:

            # ped and robot tensors
            ped_pos = get_social_agents_tensor(self.agents_history)
            robot_pos = get_odom_tensor(self.odom_history)

            # mask tensors
            ped_mask = get_mask_tensor(
                self.sequence_length, len(self.agents_history.keys()), True
            )
            veh_mask = get_mask_tensor(self.sequence_length, 1, False)

            # empty tensors
            veh_pos = torch.zeros(self.sequence_length, 1, 5)
            robot_plan_env = torch.zeros(1, 1, 1)

            ob_ped_pos, ob_ped_mask, col_ind_pres_peds = filter_curr_ob(
                ped_pos, ped_mask
            )
            ob_veh_pos, ob_veh_mask, col_ind_pres_vehs = filter_curr_ob(
                veh_pos, veh_mask
            )

            pred_pos = torch.zeros((self.pred_len, self.max_human_num, 5))
            pred_dist = torch.zeros((self.pred_len, self.max_human_num, 5))
            pred_cov = (
                torch.eye(2)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.pred_len, self.max_human_num, 1, 1)
            )
            # ones for making the inverse possible for not existing peds

            # print("=====================")
            # print(ob_ped_pos)

            with torch.no_grad():
                # ped_pred: (pred_seq_len, num_peds, 5)
                ped_pred, dist_param = self.ped_traj_pred.forward(
                    ob_ped_pos.cpu(),
                    ob_ped_mask.cpu(),
                    ob_veh_pos.cpu(),
                    ob_veh_mask.cpu(),
                    robot_pos.cpu(),
                    robot_plan_env.cpu(),
                    self.dataloader.timestamp,
                )

                mux, muy, sx, sy, corr = getCoef(dist_param.cpu())
                scaled_param_dist = torch.stack((mux, muy, sx, sy, corr), 2)
                cov = cov_mat_generation(scaled_param_dist)

                pred_pos[:, col_ind_pres_peds, :] = ped_pred.cpu()
                pred_dist[:, col_ind_pres_peds, :] = dist_param.cpu()
                pred_cov[:, col_ind_pres_peds, :, :] = cov.cpu()

                print(pred_pos)

    def recording_callback(self):
        # AGENTS RECORDING
        if self.agents_data:
            for agent in self.agents_data:
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
        if self.odom_data:
            yaw = euler_from_quaternion(
                [
                    self.odom_data.pose.pose.orientation.x,
                    self.odom_data.pose.pose.orientation.y,
                    self.odom_data.pose.pose.orientation.z,
                    self.odom_data.pose.pose.orientation.w,
                ]
            )[2]

            odom_data = (
                self.odom_data.pose.pose.position.x,
                self.odom_data.pose.pose.position.y,
                self.odom_data.twist.twist.linear.x * math.cos(yaw),
                self.odom_data.twist.twist.linear.x * math.sin(yaw),
                self.odom_data.header.stamp.sec
                + self.odom_data.header.stamp.nanosec * 1e-9,
            )
            self.odom_history[0].append(odom_data)

    def agent_states_callback(self, msg: AgentStates):
        self.agents_data = msg.agent_states

    def odom_callback(self, msg: Odometry):
        self.odom_data = msg


def main(args=None):
    rclpy.init(args=args)
    node = HumanTrajPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
