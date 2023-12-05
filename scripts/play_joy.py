import isaacgym

assert isaacgym
import torch

import glob
import pickle as pkl

from a1_gym.envs import *
from a1_gym.envs.base.legged_robot_config import Cfg
from a1_gym.envs.a1.a1_config import config_a1
from a1_gym.envs.a1.velocity_tracking import VelocityTrackingEasyEnv

import sys, getopt #读取参数

import pygame
import pygame.joystick as pyjoystick

import threading
import signal
import time
global threadExit
threadExit = False
def sig_handler(signum, frame):
    threadExit = True
    sys.exit(0)

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit').to('cuda:0')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit').to('cuda:0')
    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cuda:0'))
        action = body.forward(torch.cat((obs["obs_history"].to('cuda:0'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"./runs/{label}/*")
    logdir = sorted(dirs)[-1]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        # print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        # print(cfg.keys())
        # print(cfg)

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"

    Cfg.asset.flip_visual_attachments = True

    from a1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from a1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy

x_vel_cmd = 0
y_vel_cmd = 0
yaw_vel_cmd = 0

def main():
    gait_name = "trotting"
    date = "2023-12-04"
    try:
        opts, args = getopt.getopt(sys.argv[1:],"-h-g:-d:",["help","gait=","date="])
    except getopt.GetoptError:
        print("[*] play_cmd.py --gait=<pronking,trotting,bounding,pacing> --date=<202x-xx-xx>")
    for opt_name,opt_value in opts:
        if opt_name in ('-h','--help'):
            print("[*] play_cmd.py --gait=<pronking,trotting,bounding,pacing> --date=<202x-xx-xx>")
            exit()
        if opt_name in ('-g','--gait'):
            gait_name = opt_value
            print("[*] Set gait to: ", gait_name)
        if opt_name in ('-d','--date'):
            date = opt_value
            print("[*] Set date to: ", date)
    
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    # to see the environment rendering, set headless=False
    headless = False
    
    label = "gait-conditioned-agility/" + date + "/train"

    env, policy = load_env(label, headless=headless)
    # print(env.p_gains)
    # print(env.d_gains)

    obs = env.reset()

    while not threadExit:
        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}
        gait = torch.tensor(gaits[gait_name])
        footswing_height_cmd = 0.12
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25
        measured_x_vels = []
        target_x_vels = []
        joint_positions = []
        joint_vel = []
        base_lin_vel = []
        base_ang_vel = []
        base_height = []

        with torch.no_grad():
            actions = policy(obs)

        # Update commands with the control inputs
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd

        obs, rew, done, info = env.step(actions)

        measured_x_vels.append(env.base_lin_vel[0, 0])
        target_x_vels.append(x_vel_cmd)
        joint_positions.append(env.dof_pos[0, :].cpu())
        joint_vel.append(env.dof_vel[0, :].cpu())
        base_lin_vel.append(env.base_lin_vel[0, :].cpu())
        base_ang_vel.append(env.base_ang_vel[0, :].cpu())
        base_height.append(env.root_states[:, 2:3].cpu())

def joy_listener():
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, key
    pygame.init()
    pygame.event.clear() # 清空事件队列
    if pyjoystick.get_count() >= 1:
        print("joy connect")
    else:
        print("joy not found")
        exit() 
    js = pyjoystick.Joystick(0) # 得到第0个手柄对象
    
    while not threadExit:
        event = pygame.event.wait() # 等待事件队列
        if event.type == pygame.JOYAXISMOTION:
            # For Xbox One
            x_axis = js.get_axis(1)
            y_axis = js.get_axis(3)
            yaw_axis = js.get_axis(0)

            # 设置摇杆死区
            deadzone = 0.2
            if abs(x_axis) < deadzone:
                x_axis = 0.0
            if abs(y_axis) < deadzone:
                y_axis = 0.0
            if abs(yaw_axis) < deadzone:
                yaw_axis = 0.0

            # 计算速度命令
            x_vel_cmd = x_axis * -3.0
            y_vel_cmd = y_axis * -3.0
            yaw_vel_cmd = yaw_axis * -3.0

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    joy_thread = threading.Thread(target=joy_listener)
    main_thread = threading.Thread(target=main)

    joy_thread.setDaemon(True)
    main_thread.setDaemon(True)

    joy_thread.start()
    main_thread.start()

    while True:
        time.sleep(1)