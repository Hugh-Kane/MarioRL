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

# import os for file managament 
import os 
# import ppo for algos 
from stable_baselines3 import PPO 
# Import Base Callback for saving models 
from stable_baselines3.common.callbacks import BaseCallback

## Preprocessing environment

# Import Frame Stacker Wrapper and Grayscaling Wrapper
# FrameStack allows for memory 
# GrayScale trims the image to process down by 66%
from gym.wrappers import GrayScaleObservation
# Import Vectorisation Wrappers 
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
#Import Matplotlib - to show the impact of framestacking

from matplotlib import pyplot as plt

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
def rando_run():
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env,SIMPLE_MOVEMENT)
    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state,reward,done,_,info = env.step(env.action_space.sample())
        env.render()
        print(info)
    env.close()

class CustomResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        # Remove the 'seed' argument if it exists
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return self.env.reset(**kwargs)
    
def model_train():
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env,SIMPLE_MOVEMENT)

    env = GrayScaleObservation(env,keep_dim=True)
    env = CustomResetWrapper(env)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env,4,channels_order='last')
    """
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
    """
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    # Setup model saving callback
    #callback = TrainAndLoggingCallback(check_freq = 100000,save_path=CHECKPOINT_DIR)

    # This is the AI model started
    # There is a trade-off with the learning model - smaller rate = faster but may not converge
    # n_steps = how many steps per game to update the neural network
    model = PPO('CnnPolicy',env, verbose=1,tensorboard_log=LOG_DIR,learning_rate=0.00001,n_steps=512)

    # Train the AI model
    # This is where the AI model starts to learn
    model.learn(total_timesteps=500000)

    model.save('BestModel500000')

def model_run():
    model = PPO.load('./BestModel50000')
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env,SIMPLE_MOVEMENT)

    env = GrayScaleObservation(env,keep_dim=True)
    #env = CustomResetWrapper(env)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env,4,channels_order='last')
    state=env.reset()
    while True:
        action, _state = model.predict(state)
        state,reward,done,info = env.step(action)
        env.render

def model_run_debug():
    model = PPO.load('./BestModel10000')
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env,SIMPLE_MOVEMENT)

    env = GrayScaleObservation(env,keep_dim=True)
    env = CustomResetWrapper(env)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env,4,channels_order='last')
    state=env.reset()
    print(model.predict(state)[0][0])

#model_run_debug()
#model_train()
rando_run()
#model_train()