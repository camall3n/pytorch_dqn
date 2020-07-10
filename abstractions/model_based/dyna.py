from collections import namedtuple
import math

import gym
import torch

from abstractions.common.pqueue import PriorityQueue
from abstractions.common.replay_buffer import Experience
from abstractions.common.utils import model_based_parser
from abstractions.model_based.model import ModelNet
from abstractions.model_free.dqn.model import DQN_MLP_model

PlanningQuery = namedtuple('PlanningQuery',
    ('action', 'state', 'inner_return', 'final_state', 'done', 'n')
)

class DynaAgent:
    def __init__(self, state_space, action_space, gamma, args):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = args.device
        self.critic = DQN_MLP_model(
            args.device, state_space.shape[0], action_space, action_space.shape[0], args.model_shape
        )
        self.model = ModelNet(args, args.device, state_space, action_space).to(device=args.device)
        self.queries = PriorityQueue(maxlen=10000, mode='max')
        self.priority_threshold = 0.05
        self.priority_decay = 0.9

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def act(self, state):
        return self.action_space.sample()

    def train(self, experience):
        torch_experience = self.torchify(experience)

        td_error = self.update_agent(torch_experience)
        self.queue_rollouts(experience, 0, td_error)
        self.update_model(torch_experience)

    def plan(self):
        query = self.queries.pop_max()
        new_state, new_reward = self.rollout(query)
        new_return = new_reward + self.gamma * query.inner_return
        simulated_experience = Experience(new_state, query.action, new_return, query.final_state, query.done)
        td_error = self.update_agent(simulated_experience, n_step=query.n+1).detach()
        self.queue_rollouts(simulated_experience, query.n+1, td_error)

    def queue_rollouts(self, experience, n, td_error):
        state, _, inner_return, final_state, done = experience
        priority = torch.abs(td_error) * self.priority_decay**n
        if priority > self.priority_threshold:
            for action in range(self.action_space.n):
                new_query = PlanningQuery(action, state, inner_return, final_state, done, n)
                self.queries.push(new_query, priority)

    def rollout(self, query):
        new_state, new_reward = self.model(query.state, query.action)
        return new_state, new_reward

    def update_agent(self, experience, n_step=1):
        q_predictions = self.critic(experience.state)
        q_acted = q_predictions.gather(dim=1, index=experience.action.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            boostrapped_value = torch.max(self.critic(experience.next_state), dim=-1)[0]
            discounted_value = (1 - experience.done) * self.gamma**n_step * boostrapped_value
            q_target = experience.reward + discounted_value
            td_error = q_target - q_acted
        loss = torch.nn.functional.smooth_l1_loss(input=q_acted, target=q_target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return td_error.detach().cpu().numpy()

    def update_model(self, experience, td_error):
        state, action, reward, next_state, _ = self._torchify(experience)
        prev_state, prev_reward = self.model(next_state, action)
        state_loss = torch.nn.functional.mse_loss(input=prev_state, target=state)
        reward_loss = torch.nn.functional.mse_loss(input=prev_reward, target=reward)
        loss = state_loss + reward_loss
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def _torchify(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).byte().to(self.device)
        experiences = states, actions, rewards, next_states, dones
        return experiences

def train_agent(args):
    env = gym.make('CartPole-v1')
    agent = DynaAgent(env.state_space, env.action_space, args.gamma, args)

    state = env.reset()
    for _ in range(args.iterations):
        for _ in range(args.interactions_per_iter):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            experience = Experience(state, action, reward, next_state, done)
            agent.train(experience)
            state = next_state if not done else env.reset()
        for _ in range(args.planning_steps_per_iter):
            agent.plan()

if __name__ == "__main__":
    args = model_based_parser.parse_args()
    train_agent(args)
