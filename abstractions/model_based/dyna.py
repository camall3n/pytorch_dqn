from collections import namedtuple
import math

import gym

from abstractions.common.replay_buffer import Experience
from abstractions.common.pqueue import PriorityQueue
from abstractions.model_based.model import ModelNet

PlanningQuery = namedtuple('PlanningQuery',
    ('action', 'state', 'inner_return', 'final_state', 'done', 'n')
)

class DynaAgent:
    def __init__(self, state_space, action_space, gamma, args):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.critic =
        self.model = ModelNet(args, args.device, state_space, action_space)
        self.queries = PriorityQueue(maxlen=10000, mode='max')
        self.priority_threshold = 0.05
        self.priority_decay = 0.9

    def act(self, state):
        return self.action_space.sample()

    def train(self, experience):
        pass

    def plan(self):
        query = self.queries.pop_max()
        new_state, new_reward = self.simulate(query)
        new_return = new_reward + self.gamma * query.inner_return
        simulated_experience = Experience(new_state, query.action, new_return, query.final_state, query.done)
        td_error = self.update_agent(simulated_experience, n_step=query.n+1)

        for action in self.propose_actions():
            new_query = PlanningQuery(
                action, new_state, new_return, query.final_state, query.done, query.n+1
            )
            self.queries.push(new_query, td_error)

    def propose_actions(self):
        return range(self.action_space.n)

    def simulate(self, query):
        new_state, new_reward = self.model(query.state, query.action)
        return new_state, new_reward

    def update_agent(self, experience, n_step=1):
        td_error =
        return 0

    def update_model(self, experience):
        pass

    def update_planner(self, query, priority):
        decayed_priority = math.fabs(priority) * (self.priority_decay**query.n)
        if decayed_priority >= self.priority_threshold:
            self.queries.push((decayed_priority, query))


def train_agent():
    n_iterations = 100
    n_interactions_per_iter = 4
    n_planning_steps_per_iter = 32

    env = gym.make('CartPole-v1')
    agent = DynaAgent(n_actions=3)

    state = env.reset()
    for iteration in range(n_iterations):
        for interaction in range(n_interactions_per_iter):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            experience = Experience(state, action, reward, next_state, done)
            if done:
                state = env.reset()
            else:
                state = next_state
            agent.update_model(experience)
            agent.update_agent(experience)
        for planning_step in range(n_planning_steps_per_iter):
            agent.plan()


train_agent()