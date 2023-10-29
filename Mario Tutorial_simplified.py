# Import game
import gym_super_mario_bros
# Import joypad
from nes_py.wrappers import JoypadSpace
# Import simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


"""
#Preprocessing step
"""
#grayscale cuts down the processing power by 66% since we don't need to process all RGB channels
from gym.wrappers import GrayScaleObservation
#import vectorization wrappers 
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
from stable_baselines3.common.env_checker import check_env


# 1.Create the base environment.
env = gym_super_mario_bros.make('SuperMarioBros-v3',apply_api_compatibility=True,render_mode="human" )
# 2.Simplify controls
env = JoypadSpace(env,SIMPLE_MOVEMENT)
# 3.Grayscale
#Without keep_dim, shape corresponds to (240, 256)
#With keep_dim, shape corresponds to (240, 256, 1)
env = GrayScaleObservation(env, keep_dim=True)
# 4.Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

state= env.reset()

state = env.step(1)[0]
print(state.shape)
plt.imshow(state)
plt.show()



#print(gym_super_mario_bros.list_envs())
