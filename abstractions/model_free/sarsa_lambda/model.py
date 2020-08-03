import random
from math import floor

import torch
from gym.spaces import Discrete
import numpy as np

from ...common.modules import MLP


class SarsaAgent:
    def __init__(self,
                 device,
                 gamma,
                 alpha,
                 warmup_period,
                 lambda_value,
                 epsilon_decay_length,
                 final_epsilon_value,
                 observation_space,
                 action_space,
                 feature_size):
        assert len(observation_space.shape) == 1, "Only supports single dimensional state spaces"
        assert isinstance(action_space, Discrete), "Must be discrete actions"
        self.input_shape = observation_space.shape[0]
        self.action_space = action_space
        self.observation_space = observation_space
        self.feature_size = feature_size


        self.gamma = gamma
        self.lambda_value = lambda_value
        self.alpha = alpha

        self.device = device
        self.warmup_period = warmup_period
        self.epsilon_decay_length = epsilon_decay_length
        self.final_epsilon_value = final_epsilon_value
        self.epsilon = 1

        # self.phi = MLP([self.input_shape, feature_size], final_activation=torch.nn.Sigmoid)
        self.phi = self.binary_features
        # self.weights = torch.rand((1, self.feature_size), dtype=torch.float32, device=self.device)
        self.weights = torch.zeros((1, self.feature_size * self.action_space.n),
                                   dtype=torch.float32,
                                   device=self.device)

        self.eligibility_trace = torch.zeros_like(self.weights, device=self.device)
        self.q_old = 0

    @staticmethod
    def binary_features(state):
        state = state.squeeze(0)
        interval = [0 for i in range(len(state))]
        buckets = [2, 2, 8, 4]
        max_range = [2, 3, 0.42, 3] # [4.8,3.4*(10**38),0.42,3.4*(10**38)]

        for i in range(len(state)):
            data = state[i]
            inter = int(floor((data + max_range[i])/(2*max_range[i]/buckets[i])))
            if inter >= buckets[i]:
                interval[i] = buckets[i]-1
            elif inter < 0:
                interval[i] = 0
            else:
                interval[i] = inter
        return interval

    def feature_function(self, state, action, done):
        if done:
            return torch.zeros_like(self.weights, dtype=torch.float32, device=self.device)
        state = torch.Tensor(state, device=self.device)
        with torch.no_grad():
            features = self.phi(state.unsqueeze(0))
        action_idx_features = np.zeros((self.action_space.n, self.feature_size), dtype=np.float32)
        action_idx_features[action] = features
        return torch.from_numpy(action_idx_features.flatten())

    def compute_qval(self, state, action, done):
        features = self.feature_function(state, action, done)
        return torch.matmul(features, self.weights.transpose(0, 1)), features

    def update_eligibility(self, features_current):
        term1 = self.gamma * self.lambda_value * self.eligibility_trace
        term3 = self.alpha * self.gamma * self.lambda_value * torch.matmul(
                features_current, self.eligibility_trace.transpose(0, 1)) * features_current
        self.eligibility_trace = term1 + features_current # - term3

    def update_weights(self, td_error, features_current, q_current):
        term1 = self.alpha * td_error * self.eligibility_trace
        term2 = self.alpha * (q_current - self.q_old) * features_current
        self.weights += term1 # - term2- term2

    def train_single_batch(self, state, next_state, action, reward, done, epsilon, writer,
            writer_step):
        q_current, features_current = self.compute_qval(state, action, False)
        next_action = self.act(next_state, done, epsilon)
        q_next, _ = self.compute_qval(next_state, next_action, done)
        q_target = reward + self.gamma * q_next

        td_error = q_target - self.q_old

        self.update_eligibility(features_current)
        self.update_weights(td_error, features_current, q_current)
        self.q_old = q_next

        # Logging
        if writer:
            writer.add_scalar('training/q_current', q_current, writer_step)
            writer.add_scalar('training/q_target', q_target, writer_step)
            writer.add_scalar('training/q_next', q_next, writer_step)
            writer.add_scalar('training/td_error', td_error, writer_step)

            writer.add_histogram('training/next_action', next_action, writer_step)
            # qvals = torch.tensor([self.compute_qval(state, action, done)[0]
            #          for action in range(self.action_space.n)])
            # writer.add_histogram('training/q_values', qvals, writer_step)

            writer.add_scalar('training/mean_eligibility_trace', self.eligibility_trace.mean(),
                    writer_step)
            writer.add_scalar('training/mean_features_current', features_current.mean(),
                    writer_step)
            writer.add_scalar('training/mean_weights', self.weights.mean(), writer_step)

        return next_action

    def reset_on_termination(self):
        self.eligibility_trace = torch.zeros_like(self.weights, device=self.device)
        self.q_old = 0

    def argmax_over_actions(self, state, done):
        action_qval_pairs = [(action, self.compute_qval(state, action, done)[0])
                     for action in range(self.action_space.n)]
        return max(action_qval_pairs, key=lambda t: t[1])

    def act(self, state, done, epsilon):
        if random.random() < epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = self.argmax_over_actions(state, done)
                assert self.action_space.contains(action)
            return action

    def set_epsilon(self, global_steps, writer):
        if global_steps < self.warmup_period:
            self.epsilon = 1
        else:
            current_epsilon_decay = 1 - (1 - self.final_epsilon_value) * (
                global_steps - self.warmup_period) / self.epsilon_decay_length

            self.epsilon = max(self.final_epsilon_value, current_epsilon_decay)

        if writer:
            writer.add_scalar('training/epsilon', self.epsilon, global_steps)
