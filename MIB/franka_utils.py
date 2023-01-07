import math

from isaacgym.torch_utils import *
import torch


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """returns horizontal rotation required to grasp cube"""
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose, damping, j_eef, num_envs, device):
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def get_action(
    action,
    rb_states,
    box_idxs,
    hand_idxs,
    down_dir,
    down_q,
    dof_pos,
    box_size,
    num_envs,
    corners,
    init_pos,
    init_rot,
    hand_restart,
    j_eef,
    device,
    damping=0.05,
):
    # get box states
    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    # get hand states
    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]

    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.11

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)

    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(
        box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)
    ).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = (
        (box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)
    ).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(
        above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5
    )

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(
        return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q))
    )

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # Deploy control based on type
    action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(
        dpose, damping, j_eef, num_envs, device
    )

    # gripper actions depend on distance between hand and box
    close_gripper = (box_dist < grasp_offset + 0.02) | gripped

    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    hand_restart = hand_restart | (box_pos[:, 2] > 0.6)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(
        close_gripper,
        torch.Tensor([[0.0, 0.0]] * num_envs).to(device),
        torch.Tensor([[0.04, 0.04]] * num_envs).to(device),
    )
    action[:, 7:9] = grip_acts

    return action
