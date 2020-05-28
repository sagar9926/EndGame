import gym
from gym import spaces
import numpy as np
from gym_game.envs.CarApp import Game
from gym.spaces.box import Box

class CustomEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self):
        self.pygame = Game()
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0, high=255, shape=(28, 28, 3), dtype=np.int)

    def reset(self):
        del self.pygame
        self.pygame = Game()
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()
