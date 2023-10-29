# Import game
import gym_super_mario_bros
# Import joypad
from nes_py.wrappers import JoypadSpace
# Import simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#print(SIMPLE_MOVEMENT)

"""
#Setuo game 
#env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human" )
env = JoypadSpace(env,SIMPLE_MOVEMENT)

print(env.action_space)
print(env.observation_space.shape)
print(env.action_space.sample())


#Creating a flag - restart or not
done = True
#Loop through each fram ein the game
for step in range(100000):
    if done:
        #Star the game
        env.reset() #restarts the game
    #env.step - passes through an action into the game.
    #env.action_space.sample() is a random action in the game
    state,reward,done,info,_ = env.step(env.action_space.sample())
    print(_)
    #Show the game on the screen
    env.render()
#Close the game
env.close()
"""


"""
#Preprocessing step
"""
#grayscale cuts down the processing power by 66% since we don't need to process all RGB channels
from gym.wrappers import GrayScaleObservation
#import vectorization wrappers 
import stable_baselines3
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
from stable_baselines3.common.env_checker import check_env


# 1.Create the base environment.
env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human" )
# 2.Simplify controls
env = JoypadSpace(env,SIMPLE_MOVEMENT)
# 3.Grayscale
#Without keep_dim, shape corresponds to (240, 256)
#With keep_dim, shape corresponds to (240, 256, 1)
env = GrayScaleObservation(env, keep_dim=True)
# 4.Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

#You can confirm that the number of RGB channels go from 3 to 1 in the below
state= env.reset()
print(state.shape)
#Output - (1, 240, 256, 1)
plt.imshow(state[0])
plt.show()

"""
state = env.step(1)[0]
print(state.shape)
plt.imshow(state)
plt.show()
"""


#print(gym_super_mario_bros.list_envs())
