# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

from model import STRAW

'''
Interact with the environment
Define loss function and train_op
'''

class Trainer:

    def __init__(self, config):
        self.config = config
        self.agent = STRAW
        self.env = config.env
        self.max_episodes = config.max_episodes
        self.render = config.render
        self.gamma = config.gamma

        self.optimizer_actor = config.optimizer_actor(config.lr)
        self.optimizer_critic = config.optimizer_critic(config.critic_lr)


    def train(self):

        n_episodes = 0

        while True:

            with tf.GradientTape() as tape:
                # 1. Generate an episode
                n_episodes += 1
                action_scores, rewards, g_ts, state_values = self.generate_episode()
                # rescale the rewards
                # rewards = rewards / 100

                # 2. Compute the returns G(N_step)
                returns = self.compute_returns_N(states, rewards, self.gamma)

                loss_actor = -tf.reduce_mean((returns - critic_values) * \
                                            tf.log(action_scores))
                loss_critic = tf.reduce_mean(tf.square(returns - critic_tensors))




                # critic_base = self.critic_val.eval(feed_dict={self.input: states})
                # # 3. Do an update
                # self.train_step_actor.run(feed_dict={
                #         self.input: states,
                #         self.actions: action_takens, 
                #         self.return_val: returns,
                #         self.critic_baseline: critic_base})
                
                # self.train_step_critic.run(feed_dict={
                #         self.input: states,
                #         self.return_val: returns})
                
                # Save data or do test
                if n_episodes % self.save_interval == 0:
                    self.saver.save(sess, self.file_path)
                    
                if n_episodes % self.test_interval == 0:
                    print(datetime.now())
                    self.test(env, n_episodes)
                
                if n_episodes >= self.max_episodes:
                    break


        # with tf.Session() as sess:
            
        #     if os.path.isfile(self.file_path + '.index'):
        #         self.saver.restore(sess, self.file_path)
        #         print('Data Restored')
        #         print(datetime.now())
        #     else:
        #         self.init.run()
        #         print('Data Initialized')
        #         print(datetime.now())
            
        #     while True:
        #         # 1. Generate an episode
        #         n_episodes += 1
        #         states, action_takens, rewards =\
        #                                self.generate_episode(env, render)
        #         rewards = rewards / 100
        #         # 2. Compute the returns G(N_step)
        #         returns = self.compute_returns_N(states, rewards, gamma)
                
        #         critic_base = self.critic_val.eval(feed_dict={self.input: states})
        #         # 3. Do an update
        #         self.train_step_actor.run(feed_dict={
        #                 self.input: states,
        #                 self.actions: action_takens, 
        #                 self.return_val: returns,
        #                 self.critic_baseline: critic_base})
                
        #         self.train_step_critic.run(feed_dict={
        #                 self.input: states,
        #                 self.return_val: returns})
                
        #         # Save data or do test
        #         if n_episodes % self.save_interval == 0:
        #             self.saver.save(sess, self.file_path)
                    
        #         if n_episodes % self.test_interval == 0:
        #             print(datetime.now())
        #             self.test(env, n_episodes)
                
        #         if n_episodes >= max_episodes:
        #             break
            
        #     print(datetime.now())
        #     print('training finished')
        #     self.saver.save(sess, self.file_path)


    def generate_episode(self):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # - a list of g_ts, indexed by time step
        actions = []
        rewards = []
        g_ts = []
        state_values = []
        state_values

        n_steps = 0

        obs = self.env.reset()

        self.agent.initialize_plans()

        if self.render:
            self.env.render()
        
        while True:
            action, g_t, state_value = self.agent.plans_update_and_sample(obs)
            
            action_value = action.numpy()
            g_t_value = g_t.numpy()
            state_value_value = state_value.numpy()

            n_steps += 1
            next_obs, reward, done, _ = self.env.step(action_value)
            if self.render:
                self.env.render()

            rewards.append(reward)
            actions.append(action_value)
            g_ts.append(g_t_value)
            state_values.append(state_value_value)

            obs = next_obs
            
#            if done or n_steps >= 500:
            if done:
                break

        # want action_scores(tensor), rewards, commit_scores(tensor), critics(tensor), critics(value)

        return actions, rewards, g_ts, state_values


    def compute_returns_N(self, states, rewards, gamma):
        # compute the return G(N_step)
        T = len(rewards)
        returns = np.zeros((T))
        
        for t in reversed(range(T)):
            
            if t + self.n >= T:
                Vend = 0
            else:
                state = states[t + self.n]
                Vend = self.critic_val.eval(feed_dict = \
                                            {self.input: state.reshape((-1,8))})
                Vend = Vend[0]
            
            signal = 0
            for k in range(self.n):
                if t+k < T:
                    reward = rewards[t+k]
                else:
                    reward = 0
                signal += (gamma**k) * reward

            returns[t] = signal + (gamma**self.n) * Vend
            
        return returns