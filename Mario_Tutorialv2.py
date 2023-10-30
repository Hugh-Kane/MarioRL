# Import game
import gym_super_mario_bros
# Import joypad
from nes_py.wrappers import JoypadSpace
# Import simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Pre-requisite: 
"""
pip install gym==0.24.1 gym_super_mario_bros==7.3.0 nes_py
Trying to match the version of gym at the time of recording
super_mario_bros version is not matching the pace of gym

Reddit user specified that 0.24.1 worked too
https://www.reddit.com/r/learnpython/comments/11bafnq/cannot_get_gym_super_mario_bros_to_work/
Nic uses version 0.19.0 for gym but having issues downloading here 

Error with gym 0.23.1

Similar to Nic - downloaded PyTorch  v1.10.1

StabeBaselines3 - downloaded but, this ends up updating all packages, as 1.3.0 is throws errors. 


"""


#print(SIMPLE_MOVEMENT)
# Movements are the following: [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,SIMPLE_MOVEMENT)
# Limits the number of movement from 250+ to 7
# This can be confirmed by checking print(env.action_space)


for step in range(10000):
    #Start the game to begin with
    if done:
        env.reset()
    # Do random actions
    state,reward,done,info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()

"""
with the latest version of Gym use the following:

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