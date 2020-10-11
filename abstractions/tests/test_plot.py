# import matplotlib
# matplotlib.use('macosx')
import matplotlib.pyplot as plt

# Importing dm_control.suite or instantiating a dm2gym environment
# causes plt.show() to crash. A fix for this bug is to switch the
# matplotlib backend (see above), but this no longer waits for the
# user to manually close any open plots. However, opening a plot
# and closing it again before importing dm_control.suite seems to
# be a workaround.

plt.plot()
plt.close()

from dm_control import suite
# import gym
# env = gym.make('dm2gym:CartpoleSwingup-v0')

plt.plot()
plt.show()
