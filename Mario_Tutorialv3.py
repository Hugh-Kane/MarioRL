# Import game
import gym_super_mario_bros
# Import joypad
from nes_py.wrappers import JoypadSpace
# Import simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Pre-requisite: 
"""
Currently works with the latest version of gym (use MarioRLv2)

"""


env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env,SIMPLE_MOVEMENT)

done = True
for step in range(10000):
    #Start the game to begin with
    if done:
        env.reset()
    # Do random actions
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    env.render()
# Close the game
env.close()



## Preprocessing environment

# Import Frame Stacker Wrapper and Grayscaling Wrapper
# FrameStack allows for memory 
# GrayScale trims the image to process down by 66%
from gym.wrappers import FrameStack, GrayScaleObservation
# Import Vectorisation Wrappers 
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
#Import Matplotlib - to show the impact of framestacking

from matplotlib import pyplot as plt
"""
"""
