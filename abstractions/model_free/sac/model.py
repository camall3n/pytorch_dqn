import os

import torch
import numpy as np

from ..markov.model import MarkovHead
from ...common.modules import GaussianPolicy, QNetwork, DeterministicPolicy, build_phi_network
from ...common.utils import hard_update, soft_update
from ...common.replay_buffer import Experience


class SAC:
    def __init__(self, input_shape, action_space, device, args):
        self.enable_markov_loss = args.enable_markov_loss

        self.gamma = args.gamma
        self.tau = args.target_moving_average
        self.alpha = args.alpha

        self.model_type = args.model_type
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = device

        if self.enable_markov_loss:
            self.encoder, output_size = build_phi_network(args, input_shape)
        else:
            self.encoder, output_size = None, None

        self.critic = QNetwork(args, input_shape,
                               action_space.shape[0],
                               args.hidden_size,
                               args.model_type,
                               args.num_frames,
                               encoder=self.encoder,
                               output_size=output_size).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr, betas=(0.9, 0.999))

        self.critic_target = QNetwork(args, input_shape,
                                      action_space.shape[0],
                                      args.hidden_size,
                                      args.model_type,
                                      args.num_frames,
                                      encoder=self.encoder,
                                      output_size=output_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.enable_markov_loss:
            self.markov_head = MarkovHead(args, args.latent_dim, action_space.shape[0])
            self.markov_loss_coef = args.markov_loss_coef

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(
                    self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(0.5, 0.999))

            self.policy = GaussianPolicy(args, input_shape,
                                         action_space.shape[0],
                                         args.hidden_size,
                                         args.model_type,
                                         args.num_frames,
                                         action_space,
                                         encoder=self.encoder,
                                         output_size=output_size,
                                         detach_encoder=args.detach_encoder).to(self.device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(input_shape,
                                              action_space.shape[0],
                                              args.hidden_size,
                                              args.model_type,
                                              args.num_frames,
                                              action_space).to(self.device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    def act(self, state, evaluate=False):
        state = torch.as_tensor(state).float().to(self.device)
        state = state.unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        minibatch = memory.sample(batch_size)
        minibatch = Experience(*minibatch)

        state_batch = torch.as_tensor(minibatch.state.astype(np.float32)).to(self.device)
        next_state_batch = torch.as_tensor(minibatch.next_state.astype(np.float32)).to(self.device)
        action_batch = torch.as_tensor(minibatch.action).float().to(self.device)
        reward_batch = torch.as_tensor(minibatch.reward).float().to(self.device)
        reward_batch = reward_batch.unsqueeze(-1)
        mask_batch = torch.as_tensor(minibatch.done).float().to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch,
                    next_state_action)
            min_qf_next_target = torch.min(qf1_next_target,
                                           qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2, rep, _ = self.critic(state_batch, action_batch, return_rep=True)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = torch.nn.functional.mse_loss(qf1, next_q_value)
        qf2_loss = torch.nn.functional.mse_loss(qf2, next_q_value)

        qf_loss = qf1_loss + qf2_loss

        if self.enable_markov_loss:
            next_rep = self.encoder(next_state_batch)
            markov_loss = self.markov_head.compute_markov_loss(rep, next_rep, action_batch)
            combined_loss = qf_loss + self.markov_loss_coef * markov_loss
        else:
            combined_loss = qf_loss

        self.critic_optim.zero_grad()
        combined_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if self.enable_markov_loss:
            alg = 'smac'
        else:
            alg = 'sac'

        if actor_path is None:
            actor_path = "models/{}_actor_{}_{}".format(alg, env_name, suffix)
        if critic_path is None:
            critic_path = "models/{}_critic_{}_{}".format(alg, env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
