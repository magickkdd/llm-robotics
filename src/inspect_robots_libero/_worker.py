"""Private stdio LIBERO worker, launched by ``LiberoEmbodiment`` as a subprocess.

Does not import ``inspect_robots``, so it can also run under a separate
``$LIBERO_PYTHON`` interpreter. Protocol: one JSON request per stdin line
(``reset`` / ``step`` / ``close``), one JSON reply per stdout line. No
credentials or network model clients.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from panda_control import compliant_target, force_setpoint_for_contact
from panda_vision import pixel_offset_to_camera_delta, propose_regions

# Imports and LIBERO diagnostics go to stderr; stdout is exclusively protocol JSON.
wire = sys.stdout
sys.stdout = sys.stderr
# LIBERO's first import asks an interactive question when its config file is
# missing; here stdin is the protocol pipe, so fail loudly instead of hanging.
_config = os.path.join(os.environ.get('LIBERO_CONFIG_PATH', os.path.expanduser('~/.libero')),
                       'config.yaml')
if not os.path.exists(_config):
    wire.write(json.dumps({'error': f'LIBERO is not initialised ({_config} missing). Run '
                           '"python -c \'import libero.libero\'" once in a terminal '
                           'and answer N (see README, Installation).'}) + '\n')
    wire.flush()
    sys.exit(1)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T
import imageio.v2 as imageio


def main():
    env = None
    video = None
    obs = None
    steps = 0
    force_feedback = False
    vision_assist = False
    compliant_control = False
    compliant_axis = 2
    compliant_target_force_n = -2.0
    compliance_requested_force_n = -2.0
    compliant_contact_threshold_n = 1.0
    compliant_gain_m_per_n = 0.0003
    force_bias_world = None
    compliance_active = False
    compliance_engaged = False
    compliance_position_correction_m = 0.0
    compliance_target_force_n = 0.0
    compliance_ramp_step = 0
    vision_pixel_offset = np.zeros(2, dtype=float)
    vision_cartesian_delta_world = np.zeros(3, dtype=float)
    vision_confidence = 0.0
    vision_track_center = None

    def force_world():
        robot = env.env.robots[0]
        sensor_rotation = np.asarray(env.env.sim.data.get_site_xmat('gripper0_ft_frame'))
        return sensor_rotation @ np.asarray(robot.ee_force, dtype=float)

    def pack():
        robot = env.env.robots[0]
        # The gripper entry is the gripper's own command state, not the measured finger
        # closure. robosuite's Panda gripper only follows the sign of 2*g-1 (it keeps
        # closing above 0.5 and keeps opening below), and upstream interpolates each
        # move_to from this state value. Measured closure stays low around a wide object
        # (about 0.27 on the soup can), so a move_to(gripper=1) started from it spent
        # several steps below 0.5 and opened the hand mid-transport. current_action is
        # [-c, c] with c=-1 fully open and c=1 fully closed.
        result = {
            'images': {k: (obs[k][::-1, ::-1] if k == 'agentview_image' else obs[k]).copy().tolist() for k in
                       ('agentview_image', 'robot0_eye_in_hand_image')},
            'eef_state': np.concatenate((robot.controller.ee_pos,
                robot.controller.ee_ori_mat[:, :2].T.reshape(-1),
                [np.clip((robot.gripper.current_action[-1] + 1.0) / 2.0, 0, 1)])).tolist(),
            'gripper_qpos': obs['robot0_gripper_qpos'].tolist(),
            'success': bool(env.check_success()), 'steps': steps,
        }
        if force_feedback:
            result['wrist_ft_wrench'] = np.concatenate(
                (np.asarray(robot.ee_force, dtype=float), np.asarray(robot.ee_torque, dtype=float))
            ).tolist()
        if vision_assist:
            result['agentview_regions'] = propose_regions(
                np.asarray(result['images']['agentview_image'], dtype=np.uint8)
            ).tolist()
            result['wrist_regions'] = propose_regions(
                np.asarray(result['images']['robot0_eye_in_hand_image'], dtype=np.uint8)
            ).tolist()
            result['vision_pixel_offset'] = vision_pixel_offset.tolist()
            result['vision_cartesian_delta_world'] = vision_cartesian_delta_world.tolist()
            result['vision_confidence'] = vision_confidence
        if compliant_control:
            result['wrist_force_delta_world'] = (force_world() - force_bias_world).tolist()
            result['compliance_active'] = compliance_active
            result['compliance_position_correction_m'] = compliance_position_correction_m
            result['compliance_target_force_n'] = compliance_target_force_n
            result['compliance_engaged'] = compliance_engaged
            result['compliance_ramp_step'] = compliance_ramp_step
        return result
    try:
        for line in sys.stdin:
            req = json.loads(line)
            try:
                if req['op'] == 'reset':
                    if env is not None:
                        env.close()
                    if video is not None:
                        video.close()
                    suite_name = req.get('suite', 'libero_10')
                    force_feedback = bool(req.get('force_feedback', False))
                    vision_assist = bool(req.get('vision_assist', False))
                    compliant_control = bool(req.get('compliant_control', False))
                    if compliant_control and not force_feedback:
                        raise ValueError('compliant_control requires force_feedback')
                    compliant_axis = ('x', 'y', 'z').index(req.get('compliant_axis', 'z'))
                    compliance_requested_force_n = float(req.get('compliant_target_force_n', -2.0))
                    compliant_target_force_n = 0.0
                    compliant_contact_threshold_n = float(
                        req.get('compliant_contact_threshold_n', 1.0)
                    )
                    compliant_gain_m_per_n = float(req.get('compliant_gain_m_per_n', 0.0003))
                    suite = benchmark.get_benchmark_dict()[suite_name]()
                    task_id = int(req['task_id'])
                    task = suite.get_task(task_id)
                    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
                        task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
                    init_state = int(req['init_state'])
                    env.seed(int(req['env_seed']))
                    env.reset()
                    obs = env.set_init_state(suite.get_task_init_states(task_id)[init_state])
                    for _ in range(10):
                        obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
                    env.env.robots[0].controller.use_delta = False
                    steps = 0
                    force_bias_world = force_world() if compliant_control else None
                    compliance_active = False
                    compliance_engaged = False
                    compliance_position_correction_m = 0.0
                    compliance_target_force_n = 0.0
                    compliance_ramp_step = 0
                    vision_pixel_offset = np.zeros(2, dtype=float)
                    vision_cartesian_delta_world = np.zeros(3, dtype=float)
                    vision_confidence = 0.0
                    vision_track_center = None
                    video = imageio.get_writer(
                        req['video'],
                        fps=20,
                        codec='libx264',
                        pixelformat='yuv420p',
                        ffmpeg_params=['-movflags', '+faststart'],
                    )
                    result = pack()
                    result['instruction'] = task.language
                elif req['op'] == 'step':
                    action = np.asarray(req['action'], dtype=float)
                    if action.shape != (10,) or not np.all(np.isfinite(action)):
                        raise ValueError('Invalid action')
                    if vision_assist:
                        robot = env.env.robots[0]
                        camera_id = env.env.sim.model.camera_name2id('robot0_eye_in_hand')
                        wrist_image = np.asarray(obs['robot0_eye_in_hand_image'], dtype=np.uint8)
                        selected_index = req.get('visual_target_index')
                        vision_pixel_offset, camera_delta, vision_confidence = (
                            pixel_offset_to_camera_delta(
                                wrist_image,
                                camera_fovy_deg=float(env.env.sim.model.cam_fovy[camera_id]),
                                target_index=selected_index,
                                previous_center=vision_track_center,
                                max_delta_m=0.002,
                            )
                        )
                        if vision_confidence > 0.0:
                            vision_track_center = np.array([
                                0.5 + vision_pixel_offset[0] / wrist_image.shape[1],
                                0.5 + vision_pixel_offset[1] / wrist_image.shape[0],
                            ])
                        camera_rotation = np.asarray(
                            env.env.sim.data.cam_xmat[camera_id]
                        ).reshape(3, 3)
                        # MuJoCo camera columns are camera-local axes in world coordinates.
                        vision_cartesian_delta_world = 0.25 * camera_rotation[:, 0] * camera_delta[0]
                        vision_cartesian_delta_world -= 0.25 * camera_rotation[:, 1] * camera_delta[1]
                        action[:3] += vision_cartesian_delta_world
                    compliance_active = False
                    compliance_position_correction_m = 0.0
                    if compliant_control:
                        robot = env.env.robots[0]
                        force_delta_world = force_world() - force_bias_world
                        compliance_target_force_n, compliance_engaged, compliance_ramp_step = (
                            force_setpoint_for_contact(
                                compliance_requested_force_n,
                                force_delta_world,
                                axis=compliant_axis,
                                contact_threshold_n=compliant_contact_threshold_n,
                                contact_latched=compliance_engaged,
                                ramp_step=compliance_ramp_step,
                            )
                        )
                        target_position, compliance_position_correction_m, compliance_active = (
                            compliant_target(
                                robot.controller.ee_pos,
                                action[:3],
                                force_delta_world,
                                axis=compliant_axis,
                                target_force_n=compliance_target_force_n,
                                contact_threshold_n=compliant_contact_threshold_n,
                                gain_m_per_n=compliant_gain_m_per_n,
                                contact_latched=compliance_engaged,
                            )
                        )
                        action[:3] = target_position
                    # Upstream uses rot6d; orthonormalize its two column vectors.
                    a = action[3:6]
                    b = action[6:9]
                    if np.linalg.norm(a) < 1e-6:
                        raise ValueError('Degenerate rot6d first column')
                    a = a / np.linalg.norm(a)
                    b = b - np.dot(a, b) * a
                    if np.linalg.norm(b) < 1e-6:
                        raise ValueError('Degenerate rot6d second column')
                    b = b / np.linalg.norm(b)
                    rotation = np.stack([a, b, np.cross(a, b)], axis=1)
                    action = np.concatenate([action[:3], T.quat2axisangle(T.mat2quat(rotation)),
                                             [2 * action[-1] - 1]])
                    obs, _, _, _ = env.step(action)
                    steps += 1
                    result = pack()
                elif req['op'] == 'close':
                    break
                else:
                    raise ValueError('Unknown operation')
                video.append_data(np.concatenate([np.asarray(result['images'][k], dtype=np.uint8)
                    for k in ('agentview_image', 'robot0_eye_in_hand_image')], axis=1))
                wire.write(json.dumps(result) + '\n')
                wire.flush()
            except Exception as exc:
                wire.write(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}) + '\n')
                wire.flush()
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()

if __name__ == '__main__':
    main()
