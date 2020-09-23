import os

import pandas as pd
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

    args.batch_size = 1024
    args.n_features = env.observation_space.shape[0]
    args.n_action_dims = env.action_space.shape[0]
    args.hidden_size = 512
    args.n_updates = 2000

    def torch_isotropic_normal(mu, std):
        """Build an isotropic torch.distributions.MultivariateNormal from mu and std vectors
        """
        mu = mu.unsqueeze(0)
        std = std.unsqueeze(0)
        cov = torch.diag_embed(std, dim1=1, dim2=2)
        normal = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
        return normal

    # Construct "true" distributions
    n_distributions = 4
    mu_list = 10*torch.rand((n_distributions, args.n_action_dims)) - 5 # [-5, 5)
    std_list = (3*torch.rand((n_distributions, args.n_action_dims)) - 1.5).exp() # exp(  [-1.5, 1.5)  )
    normals = [torch_isotropic_normal(mu, std) for (mu, std) in zip(mu_list, std_list)]
    z_list = [torch.randn(1, args.n_features) for _ in normals]

    def get_batch(normals, z_list):
        a_samples = []
        z_samples = []
        n_samples = 0
        for normal, z in zip(normals, z_list):
            n_to_sample = min(args.batch_size-n_samples, args.batch_size//len(normals))
            a_samples.append(normal.sample((n_to_sample,)).squeeze(1))
            z_samples.append(z.expand(n_to_sample, args.n_features))
            n_samples += n_to_sample
        z = torch.cat(z_samples)
        a = torch.cat(a_samples)
        return z, a

    markov_head = MarkovHead(args, args.n_features, args.n_action_dims)
    optimizer = torch.optim.Adam(markov_head.parameters())

    losses = []
    for _ in tqdm(range(args.n_updates)):
        z, a = get_batch(normals, z_list)
        loss = markov_head.compute_markov_loss(z, z, a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = loss.detach().item()
        losses.append(x)

    data = pd.DataFrame()
    for dim in range(args.n_action_dims):
        for distr, (true_mu, true_std, z) in enumerate(zip(mu_list, std_list, z_list)):
            a_mu, a_std = markov_head.inverse_model(z[0], z[0])
            a_mu = a_mu.detach().numpy()
            a_std = a_std.detach().numpy()
            x = np.linspace(-10, 10, num=1000)
            p = stats.norm(loc=true_mu[dim], scale=true_std[dim]).pdf(x)
            q = stats.norm(loc=a_mu[dim], scale=a_std[dim]).pdf(x)
            data = data.append(pd.DataFrame({'x': x, 'Pr': p, 'dim': dim, 'distr': distr, 'name': 'p'}))
            data = data.append(pd.DataFrame({'x': x, 'Pr': q, 'dim': dim, 'distr': distr, 'name': 'q'}))

    plot_title = env_name.replace('dm2gym:','')
    results_dir = os.path.join('results', plot_title)
    os.makedirs(results_dir, exist_ok=True)

    with sns.plotting_context('notebook', font_scale=1.5):
        g = sns.relplot(kind='line', data=data, row='dim', col='distr', x='x', y='Pr', hue='name', style='name')
        g.fig.suptitle(plot_title, x=0.5, y=0.98)
        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.06, right=0.95)
        plt.savefig(os.path.join(results_dir, 'test_inv_model_dist.png'))
        plt.close()

    plt.plot(np.arange(len(losses)), np.asarray(losses))
    plt.title(plot_title)
    plt.ylabel('Loss')
    plt.xlabel('Updates')
    plt.savefig(os.path.join(results_dir, 'test_inv_model_loss.png'))
    plt.close()

if __name__ == "__main__":
    main()