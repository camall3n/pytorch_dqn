import time
from tqdm import tqdm

from ..common.utils import initialize_environment
from ..common.parsers import sac_parser

args = sac_parser.parse_args()

env, _ = initialize_environment(args)
env.reset()
trials = 10000
s = time.time()
for i in tqdm(range(trials)):
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
e = time.time()

print("Speed: {}".format(trials / (s - e)))
