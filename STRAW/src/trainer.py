# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

from model import STRAW

tf.enable_eager_execution()

'''
Interact with the environment
Define loss function and train_op
'''

class Trainer:

    def __init__(self, config):
        self.config = config
        self.agent = STRAW(config)
        self.env = config.env
        self.max_episodes = config.max_episodes
        self.render_when_train = config.render_when_train
        self.render_when_test = config.render_when_test
        self.gamma = config.gamma

        self.optimizer_actor = config.optimizer_actor(config.lr_actor)
        self.optimizer_critic = config.optimizer_critic(config.lr_critic)

        self.save_interval = config.save_interval
        self.test_interval = config.test_interval

        # lists to store test results
        self.test_means = []
        self.test_stds = []
        self.test_episodes = []
        self.n_test_episodes = config.n_test_episodes

        self.weights_path = config.weights_path
        self.pic_path = config.pic_path


    def train(self):

        n_episodes = 0

        if exist weights:
            self.agent.load_weights(self.weights_path)
            print('Data Restored')
            print(datetime.now())
        else:
            print('Data Initialized')
            print(datetime.now())

        while True:

            with tf.GradientTape() as tape:
                # 1. Generate an episode
                n_episodes += 1
                rewards, action_scores, state_value_tensors, state_values, commit_scores \
                                                 = self.generate_episode(self.render_when_train)

                # 2. Compute the returns G(N_step)
                # returns = self.compute_returns_N(state_values, rewards)
                returns = self.compute_returns(rewards)

                loss_actor = -tf.reduce_mean((returns - state_values) * tf.log(action_scores)) + \
                                                    self.commit_lambda*tf.reduce_mean(commit_scores)
                loss_critic = tf.reduce_mean(tf.square(returns - state_value_tensors))

            grads_actor = tape.gradient(loss_actor, self.agent.variables)
            grads_critic = tape.gradient(loss_critic, self.agent.variables)

            self.optimizer_actor.apply_gradients(zip(grads_actor, self.agent.variables))
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.agent.variables))
            
            # Save data or do test
            if n_episodes % self.save_interval == 0:
                self.agent.save_weights(self.weights_path)
                
            if n_episodes % self.test_interval == 0:
                print(datetime.now())
                self.test(n_episodes)
            
            if n_episodes >= self.max_episodes:
                break

        print(datetime.now())
        print('training finished')
        self.agent.save_weights(self.weights_path)


    def generate_episode(self, render):
        # Generates an episode by executing the current policy in the given env.
        rewards = []
        action_scores = []
        state_value_tensors = []
        state_values = []
        commit_scores = []

        n_steps = 0

        obs = self.env.reset()

        self.agent.initialize_plans()

        if render:
            self.env.render()
        
        while True:
            action, action_score, state_value_tensor, state_value, commit_score = self.agent(obs)
            
            n_steps += 1
            next_obs, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()

            rewards.append(reward)
            action_scores.append(action_score)
            state_value_tensors.append(state_value_tensor)
            state_values.append(state_value)
            commit_scores.append(commit_score)

            obs = next_obs
            
            # if done or n_steps >= 500:
            if done:
                break

        return rewards, action_scores, state_value_tensors, state_values, commit_scores


    def compute_returns(self, rewards):
        # compute the return G
        T = len(rewards)
        returns = np.zeros((T))
        return_G = 0
        
        for t in reversed(range(T)):
            return_G = rewards[t] + self.gamma * return_G
            returns[t] = return_G
            
        return returns


    def compute_returns_N(self, state_values, rewards):
        # compute the return G(N_step)
        T = len(rewards)
        returns = np.zeros((T))
        
        for t in reversed(range(T)):
            
            if t + self.N_compute_returns >= T:
                Vend = 0
            else:
                # state = states[t + self.N_compute_returns]
                # Vend = self.critic_val.eval(feed_dict = \
                #                             {self.input: state.reshape((-1,8))})
                # Vend = Vend[0]
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
        for i in range(self.n_test_episodes):
            rewards, _, _, _, _ = self.generate_episode(self.render_when_test)
            total_rewards.append(sum(rewards))
            
        total_rewards = np.array(total_rewards)
        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)
        
        print('episodes completed:', n_episodes)
        print('test mean over {} episodes:'.format(self.n_test_episodes), reward_mean)
        print('test std:', reward_std)
        print('')
        
        self.test_episodes.append(n_episodes)
        self.test_means.append(reward_mean)
        self.test_stds.append(reward_std)


    def plot_test_reward(self):
        plt.figure()
        
        plt.errorbar(self.test_episodes, self.test_means, yerr = self.test_stds)
        
        plt.title("total reward vs. the number of training episodes")
        plt.savefig(self.pic_path + "/error_bars.png")
        plt.clf()