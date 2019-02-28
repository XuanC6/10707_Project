"""
Train an agent using a PPO2 based on OpenAI Baselines.
Note: I don't clearn
"""

import time
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
from collections import defaultdict
import setup_utils, policies, ppo2
import gym
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import  make_vec_env
import sys
# sys.path.append("/Users/wangdong/Dropbox/Courses/10707/project/coinrun/coinrun")
def main():
    print("line 1...")
    args = setup_utils.setup_and_load()
    print("passed line 1...")
    comm = MPI.COMM_WORLD
    print("passed line 2...")
    rank = comm.Get_rank() #the rank of the process in a communicator
    print("passed line 3...")
    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)
    print("passed line 4,5...")
    # utils.setup_mpi_gpus()
    print("passed line 6...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    # nenvs = Config.NUM_ENVS
    nenvs = 1 #set to 1 temporarily
    # frame_stack_size = Config.FRAME_STACK_SIZE
    frame_stack_size = 1
    total_timesteps = int(5e6)
    save_interval = args.save_interval

    env_id = "MsPacman-v0"

    #copy from https://github.com/openai/baselines/blob/52255beda5f5c8760b0ae1f676aa656bb1a61f80/baselines/run.py#L33
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        # TODO: solve this with regexes
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    # env = make_vec_env(env_id, env_type, nenvs, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
    env = make_vec_env(env_id, env_type, nenvs, seed)
    env = VecFrameStack(env, frame_stack_size)

    # env = utils.make_general_env(nenvs, seed=rank)

    with tf.Session(config=config):
        # env = wrappers.add_final_wrappers(env) #don't use wrappers anymore
        
        policy = policies.get_policy()

        # ppo2.learn(policy=policy,
        #             env=env,
        #             save_interval=save_interval,
        #             nsteps=Config.NUM_STEPS,
        #             nminibatches=Config.NUM_MINIBATCHES,
        #             lam=0.95,
        #             gamma=Config.GAMMA,
        #             noptepochs=Config.PPO_EPOCHS,
        #             log_interval=1,
        #             ent_coef=Config.ENTROPY_COEFF,
        #             lr=lambda f : f * Config.LEARNING_RATE,
        #             cliprange=lambda f : f * 0.2,
        #             total_timesteps=total_timesteps)
        ppo2.learn(policy=policy,
                    env=env,
                    save_interval=save_interval,
                    nsteps=int(1e6),
                    nminibatches=100,
                    lam=0.95,
                    gamma=0.9,
                    noptepochs=16,
                    log_interval=1,
                    ent_coef=0.1,
                    lr=lambda f : f * 3e-4,
                    cliprange=lambda f : f * 0.2,
                    total_timesteps=total_timesteps)

if __name__ == '__main__':
    main()

