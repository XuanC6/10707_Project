# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_3 import A2C

tf.enable_eager_execution()

'''
Interact with the environment
Define loss function and train_op
'''

class Trainer:

    def __init__(self, config):
        self.config = config

        self.agent = A2C(config)

        self.env = config.env
        self.max_episodes = config.max_episodes
        self.render_when_train = config.render_when_train
        self.render_when_test = config.render_when_test
        self.gamma = config.gamma
        self.N_compute_returns = config.N_compute_returns

        self.optimizer = config.optimizer

        self.save_interval = config.save_interval
        self.test_interval = config.test_interval

        # lists to store test results
        self.test_episodes = []
        self.test_reward_means = []
        self.test_reward_stds = []

        self.test_lifetime_means = []
        self.test_lifetime_stds = []

        self.n_test_episodes = config.n_test_episodes

        self.weight_path = config.weight_path
        self.pic_dir = config.pic_dir


    def train(self):
        n_episodes = 0

        if os.path.isfile(self.weight_path + '.index') and self.config.restore:
            self.agent.load_weights(self.weight_path)
            print('Weights Restored')
        else:
            print('Weights Initialized')

        print(datetime.now())
        while True:

            with tf.GradientTape(persistent= True) as tape:
                # 1. Generate an episode
                n_episodes += 1
                rewards, action_scores, state_value_tensors, state_values, entropys, _ = \
                                                self.generate_episode(self.render_when_train)

                # convert to tensors
                action_scores = tf.identity(action_scores)
                state_value_tensors = tf.identity(state_value_tensors)
                entropys = tf.identity(entropys)

                # 2. Compute the returns G
                # returns = self.compute_returns(rewards)
                
                # print(np.amax(rewards), np.amin(rewards))
                rewards /= 10.0

                returns = self.compute_returns_N(rewards, state_values)

                loss_actor = -tf.reduce_mean((returns - state_values) * tf.log(action_scores)) 
                loss_critic =  tf.reduce_mean(tf.square(returns - state_value_tensors))
                
                entropy_mean = tf.reduce_mean(entropys)
                loss_entropy = -self.config.entropy_coeff * entropy_mean
                # loss_entropy = -1.0/n_episodes * entropy_mean

                loss = loss_actor + loss_critic + loss_entropy
                # loss = loss_actor + loss_critic
            
            grads = tape.gradient(loss, self.agent.variables)
            self.optimizer.apply_gradients(zip(grads, self.agent.variables))
            
            print("Episode ", n_episodes, '  ', (loss_actor.numpy(), loss_critic.numpy(), entropy_mean.numpy()))
            
            # Save data or do test
            if n_episodes % self.save_interval == 0:
                self.agent.save_weights(self.weight_path)
                
            if n_episodes % self.test_interval == 0:
                print(datetime.now())
                self.test(n_episodes)
            
            if n_episodes >= self.max_episodes:
                break

        print(datetime.now())
        print('training finished')
        self.agent.save_weights(self.weight_path)


    def generate_episode(self, render, train = True):
        # Generates an episode.
        rewards = []
        action_scores = []
        state_value_tensors = []
        state_values = []
        entropys = []

        n_steps = 0

        # reset the environment and agent
        obs = self.env.reset()

        if render:
            self.env.render()
        
        while True:
            obs = self.preprocess_observation(obs)

            # plt.imshow(obs)
            # plt.show()

            self.agent([obs])

            if train:
                action = self.agent.action_tensor.numpy()
            else:
                action = np.argmax(np.squeeze(self.agent.actions.numpy()))

            # print(action)
            # print(np.squeeze(self.agent.actions.numpy()))

            action_onehot = tf.one_hot([action], self.config.n_actions)
            action_score = tf.reduce_sum(self.agent.actions*action_onehot)
            state_value_tensor = self.agent.state_value_tensor
            state_value = state_value_tensor.numpy()
            entropy_tensor = self.agent.actions_entropy

            n_steps += 1
            next_obs, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()

            rewards.append(reward)
            action_scores.append(action_score)
            state_value_tensors.append(state_value_tensor)
            state_values.append(state_value)
            entropys.append(entropy_tensor)

            obs = next_obs
            
            # if done or n_steps >= 500:
            if done:
                break

        return np.asarray(rewards), action_scores, state_value_tensors, \
                                np.asarray(state_values), entropys, n_steps


    def compute_returns(self, rewards):
        # compute the return G
        T = len(rewards)
        returns = np.zeros((T))
        return_G = 0
        
        for t in reversed(range(T)):
            return_G = rewards[t] + self.gamma * return_G
            returns[t] = return_G
            
        return returns


    def compute_returns_N(self, rewards, state_values):
        # compute the return G(N_step)
        T = len(rewards)
        returns = np.zeros((T))
        
        for t in reversed(range(T)):
            
            if t + self.N_compute_returns >= T:
                Vend = 0
            else:
                Vend = state_values[t + self.N_compute_returns]

            signal = 0
            for k in range(self.N_compute_returns):
                if t+k < T:
                    reward = rewards[t+k]
                else:
                    reward = 0
                signal += (self.gamma**k) * reward

            returns[t] = signal + (self.gamma**self.N_compute_returns) * Vend
            
        return returns


    def test(self, n_episodes):
        # run certain test episodes on current policy, 
        # recording the mean/std of the cumulative reward.
        total_rewards = []
        lifetimes = []
        for _ in range(self.n_test_episodes):
            rewards, _, _, _, _, n_steps = self.generate_episode(self.render_when_test, train = False)
            total_rewards.append(np.sum(rewards))
            lifetimes.append(n_steps)
            
        total_rewards = np.asarray(total_rewards)
        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)

        lifetimes = np.asarray(lifetimes)
        lifetime_mean = np.mean(lifetimes)
        lifetime_std =  np.std(lifetimes)
        
        print('episodes completed:', n_episodes)
        print('test reward mean over {} episodes:'.format(self.n_test_episodes), reward_mean)
        print('test reward std:', reward_std)

        print('test lifetime mean over {} episodes:'.format(self.n_test_episodes), lifetime_mean)
        print('test lifetime std:', lifetime_std)
        print('')
        
        self.test_episodes.append(n_episodes)
        self.test_reward_means.append(reward_mean)
        self.test_reward_stds.append(reward_std)

        self.test_lifetime_means.append(lifetime_mean)
        self.test_lifetime_stds.append(lifetime_std)


    # def test(self, n_episodes):
    #     # run certain test episodes on current policy, 
    #     # recording the mean/std of the cumulative reward.
    #     mean_rewards = []
    #     for _ in range(self.n_test_episodes):
    #         rewards, _, _, _, _ = self.generate_episode(self.render_when_test)
    #         mean_rewards.append(np.mean(rewards))
            
    #     mean_rewards = np.array(mean_rewards)
    #     reward_mean = np.mean(mean_rewards)
    #     reward_std = np.std(mean_rewards)
        
    #     print('episodes completed:', n_episodes)
    #     print('test mean over {} episodes:'.format(self.n_test_episodes), reward_mean)
    #     print('test std:', reward_std)
    #     print('')
        
    #     self.test_episodes.append(n_episodes)
    #     self.test_means.append(reward_mean)
    #     self.test_stds.append(reward_std)


    def plot_test_result(self):
        plt.figure()
        plt.errorbar(self.test_episodes, self.test_reward_means, yerr = self.test_reward_stds)
        
        plt.title("total reward vs. the number of training episodes")
        plt.savefig(self.pic_dir + "/reward_error_bars.png")
        plt.clf()

        plt.figure()
        plt.errorbar(self.test_episodes, self.test_lifetime_means, yerr = self.test_lifetime_stds)
        
        plt.title("lifetime vs. the number of training episodes")
        plt.savefig(self.pic_dir + "/lifetime_error_bars.png")
        plt.clf()


    def preprocess_observation(self, obs):
        # crop
        obs = obs[1:171] 
        obs = obs.astype(np.float32)
        # normalize from -1. to 1.
        obs = (obs - 128) / 128.0
        return obs


    # def preprocess_observation(self, obs):
    #     mspacman_color = (210+164+74)/3.0
    #     # crop and downsize
    #     img = obs[1:176:2, ::2]
    #     img = img.astype(np.float32)
    #     # to grayscale
    #     img = img.mean(axis = 2) 
    #     # improve contrast
    #     img[img == mspacman_color] = 0.0 
    #     # normalize from -128 to 127
    #     img = (img - 128)/128.0
    #     return img.reshape(88, 80, 1)


