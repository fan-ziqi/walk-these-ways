import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from a1_gym.envs import *
from a1_gym.envs.base.legged_robot_config import Cfg
from a1_gym.envs.a1.a1_config import config_a1
from a1_gym.envs.a1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

import sys, getopt #读取参数

import keyboard
import threading
import sys, select, termios, tty
settings = termios.tcgetattr(sys.stdin)
key = ''
def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

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

def main(argv):
    gait_name = "trotting"
    date = "2023-12-04"
    try:
        opts, args = getopt.getopt(argv,"-h-g:-d:",["help","gait=","date="])
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

    from ml_logger import logger

    from pathlib import Path
    from a1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/" + date + "/train"

    env, policy = load_env(label, headless=headless)
    # print(env.p_gains)
    # print(env.d_gains)

    obs = env.reset()

    while True:
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

def keyboard_listener():
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, key
    try:
        print("Reading from the keyboard!")
        while key != '\x03':  # CTRL-C
            key = getKey()
            if key == 'w':
                x_vel_cmd += 0.1
            elif key == 's':
                x_vel_cmd -= 0.1
            elif key == 'a':
                y_vel_cmd += 0.1
            elif key == 'd':
                y_vel_cmd -= 0.1
            elif key == 'z':
                yaw_vel_cmd += 0.1
            elif key == 'c':
                yaw_vel_cmd -= 0.1
            else:
                x_vel_cmd = 0
                y_vel_cmd = 0
                yaw_vel_cmd = 0
                if key == '\x03':
                    break
            print(str(x_vel_cmd) + " " + str(y_vel_cmd) + " " + str(yaw_vel_cmd) )

    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.start()  # 启动键盘事件监听线程
    main(sys.argv[1:])

