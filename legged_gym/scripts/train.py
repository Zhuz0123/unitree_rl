import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    print("开始训练，将使用改进的姿态记录方法...")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
    
    
    # 训练结束后保存姿态历史
    print("训练完成，正在保存姿态历史...")
    # 选择保存几个环境的数据
    saved_paths = []
    for env_id in range(min(5, env.num_envs)):
        path = env.save_attitude_history(env_id=env_id)
        saved_paths.append(path)
    print(f"姿态历史保存完成，保存路径: {saved_paths}")
    
    # 完成训练后保存高度历史数据
    # 默认保存环境0的数据
    # save_path = env.save_height_history(env_id=0)
    # print(f"Height history saved to: {save_path}")
    
    print("可以使用以下命令查看姿态数据:")
    for path in saved_paths:
        if path:
            print(f"python legged_gym/scripts/visualize_attitude.py {path}")
            
            

if __name__ == '__main__':
    args = get_args()
    train(args)
