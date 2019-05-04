import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})

def plot_replan():
    plt.figure(figsize=(16,9))
    df_full = pd.read_csv('full_model/logs/train_log.csv')
    df_dec = pd.read_csv('Decoder_Result/logs/train_log.csv')

    plt.plot(df_full['episode'], df_full['num_steps'] / df_full['replan_times'], label='Encoder-Decoder')
    plt.plot(df_dec['episode'], df_dec['num_steps'] / df_dec['replan_times'], label='Decoder')
    plt.legend()
    plt.xlabel('Training Episodes')
    plt.ylabel('(Environment Steps) / (Replanning Steps)')
    plt.show()


def plot_crash_replan():
    plt.figure(figsize=(16,9))
    df_crash = pd.read_csv('crash_model/logs/train_log.csv')

    plt.plot(df_crash['episode'], df_crash['num_steps'] / df_crash['replan_times'], label='Encoder-Decoder')
    plt.xlabel('Training Episodes')
    plt.ylabel('(Environment Steps) / (Replanning Steps)')
    plt.show()


def plot_crash_reward():
    plt.figure(figsize=(16,9))
    eval = np.load('crash_model/logs/eval_logs.npz')

    plt.plot(eval['test_episodes'], eval['test_reward_means'])
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward per Episode')
    plt.show()
