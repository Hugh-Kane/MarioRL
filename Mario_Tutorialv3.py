# Import game
import gym_super_mario_bros
# Import joypad
from nes_py.wrappers import JoypadSpace
# Import simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

# Pre-requisite: 
"""
Currently works with the latest version of gym (use MarioRLv2)
# Name                    Version                   Build  Channel

gym                       0.26.2                   pypi_0    pypi
gym-super-mario-bros      7.4.0                    pypi_0    pypi
gymnasium                 0.28.1                   pypi_0    pypi
pillow                    10.0.0                   pypi_0    pypi
torch                     2.0.1                    pypi_0    pypi
"""





## Preprocessing environment

# Import Frame Stacker Wrapper and Grayscaling Wrapper
# FrameStack allows for memory 
# GrayScale trims the image to process down by 66%
from gym.wrappers import GrayScaleObservation
# Import Vectorisation Wrappers 
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
#Import Matplotlib - to show the impact of framestacking

from matplotlib import pyplot as plt

class CustomResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        # Remove the 'seed' argument if it exists
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return self.env.reset(**kwargs)


env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env,SIMPLE_MOVEMENT)

env = GrayScaleObservation(env,keep_dim=True)
env = CustomResetWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env,4,channels_order='last')

state =env.reset()
print(state.shape)
for _ in range(4):  # Or however many frames you are stacking
    state, _, _, _ = env.step([5])

# When you are using a vectorized environment, such as those provided by DummyVecEnv or other wrappers from stable_baselines3, you need to pass the actions as a list (or an array), even if you are only using a single environment. This is because vectorized environments are designed to handle multiple environments at once, and they expect the actions to be passed in a batched form. For example:

plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])


plt.show()