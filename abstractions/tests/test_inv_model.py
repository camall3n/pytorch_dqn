import os

import numpy as np
from scipy import stats
import seaborn as sns
import torch
from tqdm import tqdm
# ---------------------------------------------------------------
# This hack prevents a matplotlib/dm_control crash on macOS.
# We open/close a plot before importing dm_control.suite, which
# in this case happens when we import gym and make a dm2gym env.
import matplotlib.pyplot as plt
plt.plot(); plt.close()
import gym
# ---------------------------------------------------------------

from ..common.modules import MarkovHead
from ..common.utils import initialize_environment, reset_seeds
from ..common.parsers import sac_parser
from ..common.gym_wrappers import ObservationDictToInfo

def main():
    args = sac_parser.parse_args()

    # env_name = 'dm2gym:CartpoleSwingup-v0'
    # env_name = 'dm2gym:FingerSpin-v0'
    env_name = 'dm2gym:CheetahRun-v0'
    env = gym.make(env_name, environment_kwargs={'flat_observation': True})
    env = ObservationDictToInfo(env, 'observations')
    print('env_name:    ', env_name)
    print('actions:     ', env.action_space.shape)
    print('observations:', env.observation_space.shape)

    # Set seeds
    reset_seeds(args.seed)

    batch_size = 1024
    n_features = env.observation_space.shape[0]
    n_action_dims = env.action_space.shape[0]

    def torch_isotropic_normal(loc, std):
        """Build an isotropic torch.distributions.MultivariateNormal from loc and std vectors
        """
        loc = loc.unsqueeze(0)
        std = std.unsqueeze(0)
        cov = torch.diag_embed(std, dim1=1, dim2=2)
        normal = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
        return normal

    true_mu = 2.5 * torch.ones(n_action_dims)
    true_std = 0.5 * torch.ones(n_action_dims)
    normal = torch_isotropic_normal(true_mu, true_std)

    markov_head = MarkovHead(args, n_features, n_action_dims)
    optimizer = torch.optim.Adam(markov_head.parameters())

    z0 = torch.randn(1, n_features).expand(batch_size, n_features)
    z1 = torch.randn(1, n_features).expand(batch_size, n_features)
    losses = []
    for _ in tqdm(range(500)):
        a = normal.sample((batch_size,)).squeeze(1)
        loss = markov_head.compute_markov_loss(z0, z1, a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = loss.detach().item()
        losses.append(x)

    plt.plot(np.arange(len(losses)), np.asarray(losses))
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/test_inv_model_loss.png')
    plt.show()

    if n_action_dims == 1:
        plt.figure()
        a_mu, a_std = markov_head.inverse_model(z0[0], z1[0])
        a_mu = a_mu.detach().numpy()
        a_std = a_std.detach().numpy()
        print('N({}, {})'.format(a_mu, a_std))
        x = np.linspace(-10,10,num=10000)
        dx = x[1]-x[0]
        p = stats.norm(loc=true_mu, scale=true_std).pdf(x)
        q = stats.norm(loc=a_mu, scale=a_std).pdf(x)
        plt.plot(x, p)
        plt.plot(x, q)
        plt.savefig('results/test_inv_model_dist.png')
        plt.show()


if __name__ == "__main__":
    main()