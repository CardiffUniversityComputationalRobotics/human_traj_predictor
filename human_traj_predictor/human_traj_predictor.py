import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
import torch
from collections import defaultdict, deque


class HumanTrajPredictor(Node):
    def __init__(self):
        super().__init__("human_traj_predictor")

        self.recording_period = 0.3

        self.agents_data = None
        self.odom_data = None

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

        self.agent_history = defaultdict(lambda: deque(maxlen=5))
        self.odom_history = defaultdict(lambda: deque(maxlen=5))

        self.recording_timer = self.create_timer(
            self.recording_period, self.recording_callback
        )

        self.timer = self.create_timer(0.5, self.get_social_agents_tensor)

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
                self.agent_history[agent_id].append(agent_data)

        if self.odom_data:
            # ODOM RECORDING
            odom_data = (
                self.odom_data.pose.pose.position.x,
                self.odom_data.pose.pose.position.y,
                self.odom_data.twist.twist.linear.x,
                self.odom_data.twist.twist.linear.y,
                self.odom_data.header.stamp.sec
                + self.odom_data.header.stamp.nanosec * 1e-9,
            )
            self.odom_history[0].append(odom_data)

    def agent_states_callback(self, msg: AgentStates):
        self.agents_data = msg.agent_states

    def odom_callback(self, msg: Odometry):
        self.odom_data = msg

    def get_social_agents_tensor(self):
        # TENSOR STRUCTURE
        # (n_env, obs_seq_len, max_num_agent, 5)
        # where 5 includes [x, y, v_x, v_y, timestep]

        all_agents = []
        for agent_id, history in self.agent_history.items():
            positions = list(history)
            # print(positions)
            while len(positions) < 5:
                positions.insert(0, positions[0])
            all_agents.append(positions)

        if all_agents:
            all_agents_tensor = torch.tensor(all_agents, dtype=torch.float32)
            print("=================================")
            print(all_agents_tensor)
            print("#################################")
        return torch.empty((0, 5, 2))

    # def get_social_agents_tensor(self):
    #     all_agents = []
    #     for agent_id, history in self.agent_history.items():
    #         positions = list(history)  # Convert deque to list
    #         while len(positions) < 5:
    #             positions.insert(
    #                 0, positions[0]
    #             )  # Pad with the first position if less than 5
    #         all_agents.append(positions)

    #     if all_agents:
    #         return torch.tensor(all_agents, dtype=torch.float32)
    #     return torch.empty((0, 5, 2))  # Return an empty tensor if no data


def main(args=None):
    rclpy.init(args=args)
    node = HumanTrajPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
