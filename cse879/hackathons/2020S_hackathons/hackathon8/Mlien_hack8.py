#!/usr/bin/env python
# coding: utf-8

# Mason Lien
# 
# Homework
# 
# Write code to extend training to multiple episodes. Then, plot the mean entropy of the action distribution using the following function over the course of 25 episodes as well as the length of each episode. If the agent works well, the episodes should get longer. RL can be very random, so don't be surprised if you get a very good or bad outcome in your first try.

# In[1]:


# We'll start with our library imports...
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
import gym                         # to setup and run RL environments
import scipy.signal                # for a specific convolution function


# In[2]:


env = gym.make('CartPole-v1')
NUM_ACTIONS = env.action_space.n
OBS_SHAPE = env.observation_space.shape


# In[3]:


network_fn = lambda shape: tf.keras.Sequential([
                            tf.keras.layers.Dense(256, activation=tf.nn.tanh),
                            tf.keras.layers.Dense(256, activation=tf.nn.tanh),
                            tf.keras.layers.Dense(shape)])

# We'll declare our two networks
policy_network = network_fn(NUM_ACTIONS)
value_network = network_fn(1)

# Reset the environment to start a new episode
state = env.reset()
input_ = np.expand_dims(state, 0)
# Evaluate the first state
print("Policy action logits:", policy_network(input_))
print("Estimated Value:", value_network(input_))


# In[4]:


VALUE_FN_ITERS = 80
POLICY_FN_ITERS = 80
KL_MARGIN = 1.2
KL_TARGET = 0.01
CLIP_RATIO = 0.2

optimizer = tf.keras.optimizers.Adam()

def discount_cumsum(x, discount):
    """
    magic from the rllab library for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]

def categorical_kl(logp0, logp1):
    """Returns average kl divergence between two batches of distributions"""
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)

def update_fn(policy_network, value_network, states, actions, rewards, gamma=0.99):
    # Calculate the difference between how the actor did and how the critic expected it to do
    # We call this difference, "advantage"
    vals = np.squeeze(value_network(states))
    deltas = rewards[:-1] + gamma * vals[1:] - vals[:-1]
    advantage = discount_cumsum(deltas, gamma)

    # Calculate the action probabilities before any updates
    action_logits = policy_network(states)
    initial_all_logp = tf.nn.log_softmax(action_logits)
    row_indices = tf.range(initial_all_logp.shape[0])
    indices = tf.transpose([row_indices, actions])
    initial_action_logp = tf.gather_nd(initial_all_logp, indices)

    # policy loss
    for _ in range(POLICY_FN_ITERS):
        with tf.GradientTape() as tape:
            # get the policy's action probabilities
            action_logits = policy_network(states)
            all_logp = tf.nn.log_softmax(action_logits)
            
            row_indices = tf.range(all_logp.shape[0])
            indices = tf.transpose([row_indices, actions])
            action_logp = tf.gather_nd(all_logp, indices)

            # decide how much to reinforce
            ratio = tf.exp(action_logp - tf.stop_gradient(initial_action_logp))
            min_adv = tf.where(advantage > 0.,
                               tf.cast((1.+CLIP_RATIO)*advantage, tf.float32), 
                               tf.cast((1.-CLIP_RATIO)*advantage, tf.float32)
                               )
            surr_adv = tf.reduce_mean(tf.minimum(ratio[:-1] * advantage, min_adv))
            pi_objective = surr_adv
            pi_loss = -1. * pi_objective

        # update policy
        grads = tape.gradient(pi_loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

        # figure out how much the policy has changed for early stopping
        kl = categorical_kl(all_logp, initial_all_logp)
        if kl > KL_MARGIN * KL_TARGET:
            break

    # value loss
    # supervised training for a fixed number of iterations
    returns = discount_cumsum(rewards, gamma)[:-1]
    for _ in range(VALUE_FN_ITERS):
        with tf.GradientTape() as tape:
            vals = value_network(states)[:-1]
            val_loss = (vals - returns)**2
        # update value function
        grads = tape.gradient(val_loss, value_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, value_network.trainable_variables))


# In[5]:


def categorical_entropy(logits):
    return -1 * tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=-1))


# In[8]:


state_buffer = []
action_buffer = []
reward_buffer = []
entropy_mean = []
action_length = []

for i in range(25):
    done = False
    state = env.reset()
    i = 0
    # Collect experiences from a one episode rollout
    while not done:
        # store the initial state (and every state thereafter)
        state_buffer.append(state)

        # choose action
        action_logits = policy_network(np.expand_dims(state, 0))
        action = np.squeeze(tf.random.categorical(action_logits, 1))
        action_buffer.append(action)

        # step environment
        state, rew, done, info = env.step(action)
        reward_buffer.append(rew)

    # Run training update
    states = np.stack(state_buffer)
    actions = np.array(action_buffer)
    rewards = np.array(reward_buffer)
    update_fn(policy_network, value_network, states, actions, rewards)
    
    entropy = categorical_entropy(states)
    entropy_mean.append(entropy)
    
    number_of_actions = len(actions)
    action_length.append(number_of_actions)


# In[9]:


plt.plot(entropy_mean)


# In[13]:


plt.plot(action_length)


# In[20]:


action_length


# In[ ]:




