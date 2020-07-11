from collections import namedtuple
import math

import gym
import numpy as np
import torch

from abstractions.common.pqueue import PriorityQueue
from abstractions.common.replay_buffer import Experience
from abstractions.common.utils import model_based_parser
from abstractions.model_based.model import ModelNet
from abstractions.model_free.dqn.model import DQN_MLP_model

Trajectory = namedtuple('Trajectory',
    ('action', 'state', 'inner_return', 'final_state', 'done', 'n')
)

class DynaAgent:
    def __init__(self, state_space, action_space, gamma, args):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = args.device
        self.critic = DQN_MLP_model(
            args.device, state_space, action_space, action_space.n, args.model_shape
        )
        self.model = ModelNet(args, args.device, state_space, action_space).to(device=args.device)
        self.queries = PriorityQueue(maxlen=10000, mode='max')
        self.priority_threshold = 0.05
        self.priority_decay = 0.9

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def act(self, state):
        return self.action_space.sample()

    def train(self, experiences):
        batch = self._torchify_experience(Experience(*list(zip(*experiences))))

        td_errors = self.update_agent(batch)
        for experience, td_error in zip(experiences, td_errors):
            self.queue_rollouts(experience, 0, td_error)
        self.update_model(batch)

    def plan(self, simulator_steps):
        query_list = []
        for _ in range(simulator_steps):
            if self.queries:
                query_list.append(self.queries.pop_max())
        trajectories = self.rollout(query_list)
        batch = Experience(
            trajectories.state,
            trajectories.action,
            trajectories.inner_return,
            trajectories.final_state,
            trajectories.done
        )
        td_errors = self.update_agent(batch, trajectories.n)
        cpu_batch = list(map(lambda x: x.detach().cpu().numpy(), batch))
        experiences = list(map(lambda x: Experience(*x), zip(*cpu_batch)))
        for experience, n, td_error in zip(experiences, trajectories.n, td_errors):
            self.queue_rollouts(experience, n.detach().cpu().numpy(), td_error)

    def queue_rollouts(self, experience, n, td_error):
        state, _, inner_return, final_state, done = experience
        priority = np.abs(td_error) * self.priority_decay**n
        if priority > self.priority_threshold:
            for action in np.arange(self.action_space.n):
                new_query = Trajectory(action, state, inner_return, final_state, done, n)
                self.queries.push(new_query, priority)

    def rollout(self, batch):
        batch = Trajectory(*list(zip(*batch)))
        batch = self._torchify_planning_query(batch)
        new_state, new_reward = self.model(batch.state, batch.action)
        new_return = new_reward.detach().squeeze(-1) + self.gamma * batch.inner_return
        simulated_experiences = Trajectory(
            batch.action,
            new_state.detach(),
            new_return,
            batch.final_state,
            batch.done,
            batch.n
        )
        return simulated_experiences

    def update_agent(self, batch, n_steps=1):
        q_predictions = self.critic(batch.state)
        q_acted = q_predictions.gather(dim=1, index=batch.action.long()).squeeze(1)
        with torch.no_grad():
            boostrapped_value = torch.max(self.critic(batch.next_state), dim=-1)[0]
            discounted_value = (1 - batch.done) * self.gamma**n_steps * boostrapped_value
            q_target = batch.reward + discounted_value
            td_error = q_target - q_acted
        loss = torch.nn.functional.smooth_l1_loss(input=q_acted, target=q_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return td_error.detach().cpu().numpy()

    def update_model(self, batch):
        state, action, reward, next_state, _ = batch
        prev_state, prev_reward = self.model(next_state, action)
        state_loss = torch.nn.functional.mse_loss(input=prev_state, target=state)
        reward_loss = torch.nn.functional.mse_loss(input=prev_reward, target=reward)
        loss = state_loss + reward_loss
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def _torchify_experience(self, batch):
        batch = map(np.stack, batch)
        states, actions, rewards, next_states, dones = batch
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device).unsqueeze(-1)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).byte().to(self.device)
        batch = Experience(states, actions, rewards, next_states, dones)
        return batch

    def _torchify_planning_query(self, batch):
        batch = list(map(np.stack, batch))
        action, state, inner_return, final_state, done, n_steps = batch
        action = torch.from_numpy(action).long().to(self.device).unsqueeze(-1)
        state = torch.from_numpy(state).float().to(self.device)
        inner_return = torch.from_numpy(inner_return).float().to(self.device)
        final_state = torch.from_numpy(final_state).float().to(self.device)
        done = torch.from_numpy(done).byte().to(self.device)
        n_steps = torch.from_numpy(n_steps).long().to(self.device)
        batch = Trajectory(action, state, inner_return, final_state, done, n_steps)
        return batch

def train_agent(args):
    env = gym.make(args.env)
    agent = DynaAgent(env.observation_space, env.action_space, args.gamma, args)

    state = env.reset()
    episode = 0
    ep_reward = 0
    for _ in range(args.iterations):
        experiences = []
        for _ in range(args.interactions_per_iter):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            experiences.append(Experience(state, action, reward, next_state, done))
            state = next_state if not done else env.reset()
            if done:
                episode+=1
                print(episode, ep_reward)
                ep_reward = 0
        agent.train(experiences)
        agent.plan(simulator_steps=args.planning_steps_per_iter)

if __name__ == "__main__":
    args = model_based_parser.parse_args()
    if args.gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    train_agent(args)
