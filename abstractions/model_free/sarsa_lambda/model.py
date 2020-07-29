import torch
import random
from gym.spaces import Discrete

from ...common.modules import MLP


class SarsaAgent:
    def __init__(self,
                 device,
                 gamma,
                 alpha,
                 lambda_value,
                 observation_space,
                 action_space,
                 feature_size):
        assert len(observation_space.shape) == 1, "Only supports single dimensional state spaces"
        assert isinstance(action_space, Discrete), "Must be discrete actions"
        self.input_shape = observation_space.shape[0] + 1
        self.action_space = action_space
        self.observation_space = observation_space
        self.feature_size = feature_size
        self.device = device

        self.gamma = gamma
        self.lambda_value = lambda_value
        self.alpha = alpha

        self.epsilon = 1

        self.phi = MLP([self.input_shape, feature_size])
        self.weights = torch.rand((self.feature_size, 1), dtype=torch.float32, device=self.device)

        self.reset_on_termination()

    def feature_function(self, state, action, done):
        state = torch.Tensor(state, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        if done:
            return torch.zeros_like(self.weights, device=self.device)
        else:
            with torch.no_grad():
                return self.phi(torch.cat([state, action]).unsqueeze(0)).transpose(0, 1)

    def compute_qval(self, state, action, done):
        features = self.feature_function(state, action, done)
        return torch.matmul(self.weights.transpose(0, 1), features), features

    def update_eligibility(self, features_current):
        self.eligibility_trace *= self.gamma * self.lambda_value
        term3 = self.alpha * self.gamma * self.lambda_value * torch.matmul(
                self.eligibility_trace.transpose(0, 1), features_current) * features_current
        self.eligibility_trace += features_current - term3

    def update_weights(self, td_error, features_current, q_current):
        term1 = self.alpha * (td_error + q_current - self.q_old) * self.eligibility_trace
        term2 = self.alpha * (q_current - self.q_old) * features_current
        self.weights += term1 - term2

    def train_single_batch(self, state, next_state, action, reward, done, epsilon):
        q_current, features_current = self.compute_qval(state, action, 0)
        next_action = self.act(state, done, epsilon)
        q_next, _ = self.compute_qval(next_state, next_action, done)

        td_error = reward + self.gamma * q_next - q_current
        self.update_eligibility(features_current)
        self.update_weights(td_error, features_current, q_current)
        return next_action

    def reset_on_termination(self):
        self.eligibility_trace = torch.zeros_like(self.weights, device=self.device)
        self.q_old = 0

    def argmax_over_actions(self, state, done):
        max_pairs = [(action, self.compute_qval(state, action, done))
                     for action in range(self.action_space.n)]
        return max(max_pairs, key=lambda t: t[1])

    def act(self, state, done, epsilon):
        if random.random() < epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.Tensor(state).unsqueeze(0)
                action_tensor, _ = self.argmax_over_actions(state_tensor, done)
                action = action_tensor.cpu().detach().numpy().flatten()[0]
                assert self.action_space.contains(action)
            return action

    def set_epsilon(self, steps, writer):
        pass
