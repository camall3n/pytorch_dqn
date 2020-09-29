import time

from ..common.utils import initialize_environment
from ..common.parsers import sac_parser

args = sac_parser.parse_args()

env, _ = initialize_environment(args)
trials = 10000
s = time.time()
for i in range(trials):
    _ = env.step(env.action_space.sample())
e = time.time()

print("Speed: {}".format(trials / (s - e)))
