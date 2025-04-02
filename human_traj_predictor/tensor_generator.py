import torch


def get_social_agents_tensor(agent_history):
    # TENSOR STRUCTURE
    # (n_env, obs_seq_len, max_num_agent, 5)
    # where 5 includes [x, y, v_x, v_y, timestep]

    all_agents = []
    for agent_id, history in agent_history.items():
        positions = list(history)
        # print(positions)
        while len(positions) < 5:
            positions.insert(0, positions[0])
        all_agents.append(positions)

    if all_agents:
        all_agents_tensor = torch.transpose(
            torch.tensor(all_agents, dtype=torch.float32), 0, 1
        )
        return all_agents_tensor
    return torch.empty((0, 5, 2))


def get_odom_tensor(odom_history):
    # TENSOR STRUCTURE
    # (n_env, obs_seq_len, 1, 5)
    # where 5 includes [x, y, v_x, v_y, timestep]

    odom_data = []
    for id_, history in odom_history.items():
        positions = list(history)  # Convert deque to list
        while len(positions) < 5:
            positions.insert(
                0, positions[0]
            )  # Pad with the first position if less than 5
        odom_data.append(positions)

    if odom_data:
        odom_tensor = torch.transpose(
            torch.tensor(odom_data, dtype=torch.float32), 0, 1
        )
        # print(odom_tensor)
        return odom_tensor
    return torch.empty((0, 5, 2))  # Return an empty tensor if no data


def get_mask_tensor(sequence_length, num_agents, mask_value):
    # TENSOR STRUCTURE
    # (n_env, obs_seq_len, max_num_agent)
    mask_tensor = torch.full((sequence_length, num_agents), mask_value)
    # print(mask_tensor.shape)
    # print(mask_tensor)
    return mask_tensor


def filter_curr_ob(pos, mask):
    """
    This function removes those columns in pos and mask
    that are assosicated to agents that are not present
    in any of the frames during this current sequence length
    that we are looking at in the scenario
    """
    num_frame = pos.shape[0]
    # columns with value of zero are associated to those peds not available in this whole sequence
    num_avail_fram = torch.sum(mask, 0)
    columns_to_keep = (num_avail_fram != 0).nonzero()  # of shape (num_valid_columns, 1)
    # expanding this valid column number to all time rows in the pos and mask (first dimension)
    columns_to_keep_rp = torch.transpose(columns_to_keep, 1, 0).repeat(num_frame, 1)
    mask_filter = torch.gather(mask, dim=1, index=columns_to_keep_rp)
    columns_to_keep_rp2 = columns_to_keep_rp.unsqueeze(2).repeat(1, 1, pos.shape[2])
    pos_filter = torch.gather(pos, dim=1, index=columns_to_keep_rp2)

    col_indexs_to_keep = columns_to_keep_rp[0, :]

    return pos_filter, mask_filter, col_indexs_to_keep
