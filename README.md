# human_traj_predictor

ROS 2 package for online human trajectory prediction from `pedsim` agent states and
robot odometry.

The package runs a Python node that records a short history of pedestrian and robot
motion, feeds that history into a trained PyTorch trajectory prediction model, and
publishes predicted future pedestrian poses with 2D covariance estimates plus RViz
markers.

## Package Contents

- `human_traj_predictor/human_traj_predictor.py`: ROS 2 node, topic interface,
  history buffering, prediction publishing, and marker publishing.
- `human_traj_predictor/tensor_generator.py`: converts ROS state history into
  tensors used by the predictor.
- `human_traj_predictor/ped_pred/`: PyTorch model code, interaction grid utilities,
  data loader, HBS data, preprocessed test data, and trained model files.
- `setup.py`: installs the node executable and packaged model/data artifacts.

## ROS 2 Interface

### Node

Executable:

```bash
ros2 run human_traj_predictor human_traj_predictor_node
```

Node name:

```text
human_traj_predictor
```

### Subscriptions

| Topic | Type | QoS depth | Purpose |
| --- | --- | --- | --- |
| `/pedsim_simulator/simulated_agents` | `pedsim_msgs/msg/AgentStates` | `1` | Current simulated pedestrian states. |
| `/odom` | `nav_msgs/msg/Odometry` | `1` | Current robot pose and velocity. |

The agent callback stores `msg.agent_states` and updates the active number of
humans to the number of agents in the latest message. The odometry callback stores
the latest robot odometry sample.

### Publishers

| Topic | Type | QoS depth | Purpose |
| --- | --- | --- | --- |
| `agents_prediction` | `tidup_move_base_msgs/msg/AgentStatesPrediction` | `10` | Predicted future poses and covariance for each observed agent. |
| `agents_pred_marker` | `visualization_msgs/msg/MarkerArray` | `10` | Red sphere markers for visualizing predicted positions in RViz. |

Publisher topic names are relative, so they resolve inside the node namespace unless
remapped.

Example remapping:

```bash
ros2 run human_traj_predictor human_traj_predictor_node --ros-args \
  -r /odom:=/robot/odom \
  -r /pedsim_simulator/simulated_agents:=/people \
  -r agents_prediction:=/people/prediction
```

## Message Types

### Input: `pedsim_msgs/msg/AgentStates`

```text
std_msgs/Header header
pedsim_msgs/AgentState[] agent_states
```

Each `AgentState` contains:

```text
std_msgs/Header header
uint64 id
uint16 type
string social_state
geometry_msgs/Pose pose
geometry_msgs/Twist twist
pedsim_msgs/AgentForce forces
```

The predictor uses each agent's:

- `id`
- `pose.position.x`
- `pose.position.y`
- `twist.linear.x`
- `twist.linear.y`
- `header.stamp`
- `header.frame_id`

### Input: `nav_msgs/msg/Odometry`

The predictor uses:

- `pose.pose.position`
- `pose.pose.orientation`
- `twist.twist.linear.x`
- `header.stamp`

The odometry orientation is converted to yaw, then the robot forward velocity is
projected into world-frame `vx` and `vy`.

### Output: `tidup_move_base_msgs/msg/AgentStatesPrediction`

```text
tidup_move_base_msgs/AgentStatePrediction[] agent_states_prediction
```

Each `AgentStatePrediction` contains:

```text
pedsim_msgs/AgentState agent_state
tidup_move_base_msgs/PoseWith2DCovariance[] predicted_poses
```

Each predicted pose contains:

```text
std_msgs/Header header
geometry_msgs/Pose pose
float64[4] covariance
```

For every agent, the node publishes:

- The current pose as the first item in `predicted_poses`, with near-zero
  covariance.
- Six future poses at `0.5 s` intervals, for a `3.0 s` prediction horizon.
- A flattened 2D covariance matrix in row-major order:
  `[cov_xx, cov_xy, cov_yx, cov_yy]`.

### Output: `visualization_msgs/msg/MarkerArray`

The marker publisher creates red `SPHERE` markers for each predicted agent position.
Markers are published in the `map` frame with a scale of `0.2 m`.

## Configuration

The current implementation does not declare ROS 2 parameters. Runtime settings are
hard-coded in `human_traj_predictor/human_traj_predictor.py` and in the packaged
model `config.pkl`.

### Node Settings

| Setting | Current value | Meaning |
| --- | --- | --- |
| `recording_period_` | `0.5` seconds | Period used to sample agent and odometry history. |
| `sequence_length_` | `6` frames | Observed history length. At `0.5 s`, this is `3.0 s`. |
| `pred_len_` | `6` frames | Number of future prediction steps. At `0.5 s`, this is `3.0 s`. |
| `max_human_num_` | Starts at `10`, then updates to latest agent count | Number of agents included in the prediction output. |
| prediction timer | `0.1` seconds | How often the node attempts to run prediction. |
| model path | `ped_pred/TrainedModel/uncertainty_aware_model/CollisionGrid` | Packaged checkpoint and config used at runtime. |
| CUDA | `True` in the node/model loader | The runtime path expects CUDA unless the code is changed. |

Prediction starts only after both agent states and odometry have been received and
more than `sequence_length_` history samples have been recorded.

### Model Config Used At Runtime

The node loads the uncertainty-aware CollisionGrid model:

| Config field | Value |
| --- | --- |
| `method` | `4` (`CollisionGrid`) |
| `obs_length` | `6` |
| `pred_length` | `6` |
| `seq_length` | `12` |
| `input_size` | `2` |
| `output_size` | `5` |
| `rnn_size` | `128` |
| `embedding_size` | `64` |
| `dropout` | `0.5` |
| `gru` | `False` |
| `num_sector` | `8` |
| pedestrian TTC thresholds | `[9]` |
| pedestrian conflict distance | `0.7 m` |
| vehicle/ego TTC thresholds | `[8]` |
| vehicle/ego conflict distance | `1.0 m` |
| `store_grid` | `True` |

The repository also includes VanillaLSTM and uncertainty-unaware model artifacts,
but `setup.py` installs and `human_traj_predictor.py` loads only the
uncertainty-aware CollisionGrid checkpoint.

## Prediction Pipeline

1. Subscribe to pedsim agent states and robot odometry.
2. Record agent and robot history every `0.5 s`.
3. Convert each history sample into tensors with:
   `[x, y, vx, vy, timestamp]`.
4. Build pedestrian masks and robot/ego tensors.
5. Filter out agents that are not present in the current observation window.
6. Convert positions into displacement form for the recurrent model.
7. Build interaction grids using time-to-conflict, distance thresholds, and
   approach-angle sectors.
8. Run the trained PyTorch CollisionGrid LSTM.
9. Interpret model outputs as bivariate Gaussian parameters:
   `[mu_x, mu_y, sigma_x, sigma_y, rho]`.
10. Use the Gaussian mean as the predicted position.
11. Convert predicted displacements back into absolute positions.
12. Convert Gaussian parameters into 2D covariance matrices.
13. Publish structured predictions and RViz markers.

## Techniques Used

- ROS 2 Python node with `rclpy`.
- Pedsim-based pedestrian state input.
- Robot odometry conditioning.
- PyTorch recurrent neural network inference.
- LSTMCell-based `CollisionGridModel`.
- Social interaction encoding with time-to-conflict grids.
- Pedestrian-pedestrian and pedestrian-ego interaction features.
- Approach-angle sectorization with `8` sectors.
- Bivariate Gaussian trajectory output.
- Covariance generation from predicted Gaussian parameters.
- Displacement-space prediction with conversion back to absolute coordinates.

There is also Kalman-filter covariance generation code in `ped_pred/helper.py`, but
that path is currently commented out in the runtime predictor.

## Build

From the root of your ROS 2 workspace:

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select human_traj_predictor
source install/setup.bash
```

This package depends on message packages that must be available in the same
workspace or sourced from another workspace:

- `pedsim_msgs`
- `tidup_move_base_msgs`

Python/runtime dependencies used by the node include:

- `rclpy`
- `numpy`
- `torch`
- `tf_transformations`
- `launch_ros`
- `nav_msgs`
- `visualization_msgs`

The current `package.xml` only declares test dependencies, so make sure the runtime
dependencies are installed even if `rosdep` does not catch all of them.

## Run

Start the simulator or data source that publishes:

- `/pedsim_simulator/simulated_agents`
- `/odom`

Then run:

```bash
ros2 run human_traj_predictor human_traj_predictor_node
```

Inspect the topic output:

```bash
ros2 topic echo /agents_prediction
ros2 topic echo /agents_pred_marker
```

In RViz, add a `MarkerArray` display and subscribe it to `/agents_pred_marker`.

## Notes and Limitations

- Topics, timing, horizon length, and CUDA use are currently hard-coded rather than
  exposed as ROS parameters.
- The node expects the trained model checkpoint and `config.pkl` to be installed in
  the package share directory.
- The runtime path uses the CollisionGrid model only.
- Non-ego vehicle tensors are currently placeholder tensors with masks set to
  `False`.
- The observed robot odometry is used as an ego interaction input. Future ego motion
  is currently represented by a zero tensor placeholder.
- The data loader uses the packaged HBS test preprocessed file at runtime for timing
  metadata. Regenerating preprocessed data from raw CSV may require restoring the
  commented pandas read path in `DataLoader.py`.
- The node publishes predictions after enough history has been collected, so the
  first output appears after several `0.5 s` recording ticks.

## Troubleshooting

- If the node cannot import custom messages, build and source the workspaces
  containing `pedsim_msgs` and `tidup_move_base_msgs`.
- If the model checkpoint is not found, rebuild the package and confirm the trained
  model artifacts were installed under `install/human_traj_predictor/share`.
- If CUDA is unavailable, change the predictor initialization and saved model
  settings to use CPU before running on a CPU-only machine.
- If no predictions are published, confirm that both input topics are active and
  that `/pedsim_simulator/simulated_agents` contains at least one agent.
