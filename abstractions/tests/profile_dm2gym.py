import time
from tqdm import tqdm

from ..common.utils import initialize_environment
from ..common.parsers import sac_parser

args = sac_parser.parse_args(args=[
    '--env',
    'Visualdm2gym:CartpoleSwingup-v0',
    '--run-tag',
    'test-sac-dm-cartpole',
    '--model-type',
    'curl',
    '--no-atari',
    '--detach-encoder',
    '--max-steps',
    '2000',
    '--enable-markov-loss',
    '--action-repeat',
    '8'
])

env, _ = initialize_environment(args)
env.reset()
trials = 1000
s = time.time()
for i in tqdm(range(trials)):
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
e = time.time()

print("Speed: {}".format(trials / (e - s)), 'iterations per second')
