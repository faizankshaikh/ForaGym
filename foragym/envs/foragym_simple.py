import gym
import numpy as np

from gym import spaces

class ForaGym(gym.Env):
    metadata = {"render_modes": ["human", "text"]}
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        self.FIELDS = 5
        self.NUM_LIFE_POINTS = 7
        self.ACTION_DICT = {0: "Wait", 1: "Forage"}
        self.WEATHER_DICT = {0: "Clear", 1: "Rainy"}
        self.NUM_ACTIONS = len(self.ACTION_DICT)

        self.is_alive = True
        self.episode_length = 6

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            "field_state": spaces.MultiBinary(self.FIELDS),
            "life_points": spaces.Discrete(7),
            "weather_type": spaces.Discrete(2)
        })

        self.P = {lp: {val: [] for val in range(self.NUM_ACTIONS)} for lp in range(self.NUM_LIFE_POINTS)}
        
    def _get_transition_probs(self):
        pass
                        
    def _get_obs(self):
        return {
            "field_state": self.field_state,
            "life_points": self.life_points,
            "weather_type": self.weather_type
        }

    def step(self, action):
        
        self.field_state = self.observation_space["field_state"].sample()
        self.weather_type = self.observation_space["weather_type"].sample()
        self.prob_success = sum(self.field_state) / len(self.field_state)
        self.prob_failure = 1 - self.prob_success
        
        obs = self._get_obs()
        
        if action:
            chance = np.random.sample()
            chance = (chance - 0.1) if self.weather_type else chance
            
            if chance >= self.prob_failure:
                reward = 1
            else:
                reward = -2
        else:
            chance = 0
            reward = -1

        self.life_points += reward
        self.life_points = np.clip(self.life_points, 0, 6)
        self.episode_length -= 1

        if self.life_points <= 0 or self.episode_length <= 0:
            self.is_alive = False
        else:
            self.is_alive = True

        return obs, reward, self.is_alive, {"chance": chance}

    def reset(self, seed=42):
        super().reset(seed=seed)
        np.random.seed(42)
        
        self.field_state = self.observation_space["field_state"].sample()
        self.life_points = self.observation_space["life_points"].sample()
        self.weather_type = self.observation_space["weather_type"].sample()
        self.prob_success = sum(self.field_state) / len(self.field_state)
        self.prob_failure = 1 - self.prob_success
        
        self.episode_length = 6
        
        obs = self._get_obs()
        self._get_transition_probs()
        
        return obs

    def render(self, mode="human", close=False):
        print(f"--Days left: {self.episode_length}")
        print(f"--State of Field: {self.field_state}")
        print(f"--Current life: {self.life_points}")
        print(f"--Type of Weather: {self.WEATHER_DICT[self.weather_type]}")
        