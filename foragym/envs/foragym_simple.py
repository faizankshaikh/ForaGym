import gym
import numpy as np
from gym import spaces


class ForaGym(gym.Env):
    metadata = {"render_modes": ["human", "text"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.ACTION_DICT = {0: "Wait", 1: "Forage"}
        self.WEATHER_DICT = {0: "Clear", 1: "Rainy"}

        self.NUM_FIELDS = 5
        self.NUM_LIFE_POINTS = 7
        self.NUM_WEATHER_TYPES = 2
        self.NUM_ACTIONS = len(self.ACTION_DICT)

        self.is_alive = True
        self.episode_length = 6

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "field_state": spaces.MultiBinary(self.NUM_FIELDS),
                "life_points": spaces.Discrete(self.NUM_LIFE_POINTS),
                "weather_type": spaces.Discrete(self.NUM_WEATHER_TYPES),
            }
        )

        self.NUM_STATES = (
            (self.NUM_FIELDS + 1) * self.NUM_LIFE_POINTS * self.NUM_WEATHER_TYPES
        )

        self.P = {
            state: {action: [] for action in range(self.NUM_ACTIONS)}
            for state in range(self.NUM_STATES)
        }

        self._get_new_day(with_life_points=True)

        self._get_transition_probs()

    def _get_new_day(self, with_life_points=True):
        self.field_state = self.observation_space["field_state"].sample()
        self.weather_type = self.observation_space["weather_type"].sample()
        self.prob_success = sum(self.field_state) / len(self.field_state)
        self.prob_failure = 1 - self.prob_success

        if with_life_points:
            self.life_points = self.observation_space["life_points"].sample()

    def _get_transition_probs(self):
        for field in range(self.NUM_FIELDS + 1):
            for life_point in range(self.NUM_LIFE_POINTS):
                for weather in range(self.NUM_WEATHER_TYPES):
                    prob_success = field / self.NUM_FIELDS
                    prob_failure = 1 - prob_success
                    enc_state = self.encode(field, life_point, weather)
                    for action in range(self.NUM_ACTIONS):
                        if not life_point:
                            self.P[enc_state][action].append([1, enc_state, 0, False])
                        elif life_point and action:
                            reward = -2
                            new_life_point_failure = np.clip(
                                life_point + reward, 0, self.NUM_LIFE_POINTS - 1
                            )
                            new_enc_state = self.encode(
                                field, new_life_point_failure, weather
                            )
                            if new_life_point_failure:
                                self.P[enc_state][action].append(
                                    [prob_failure, new_enc_state, reward, True]
                                )
                            else:
                                self.P[enc_state][action].append(
                                    [prob_failure, new_enc_state, reward, False]
                                )
                            reward = 1
                            new_life_point_success = np.clip(
                                life_point + reward, 0, self.NUM_LIFE_POINTS - 1
                            )
                            new_enc_state = self.encode(
                                field, new_life_point_success, weather
                            )
                            self.P[enc_state][action].append(
                                [prob_success, new_enc_state, reward, True]
                            )
                        else:
                            reward = -1
                            new_life_point_wait = np.clip(
                                life_point + reward, 0, self.NUM_LIFE_POINTS - 1
                            )
                            new_enc_state = self.encode(
                                field, new_life_point_wait, weather
                            )
                            if new_life_point_wait:
                                self.P[enc_state][action].append(
                                    [1.0, new_enc_state, reward, True]
                                )
                            else:
                                self.P[enc_state][action].append(
                                    [1.0, new_enc_state, reward, False]
                                )

    def encode(self, field, life_point, weather):
        enc_state = field

        enc_state *= self.NUM_LIFE_POINTS
        enc_state += life_point

        enc_state *= self.NUM_WEATHER_TYPES
        enc_state += weather

        return enc_state

    def decode(self, enc_state):
        out = []
        out.append(enc_state % self.NUM_WEATHER_TYPES)
        enc_state = enc_state // self.NUM_WEATHER_TYPES

        out.append(enc_state % self.NUM_LIFE_POINTS)
        enc_state = enc_state // self.NUM_LIFE_POINTS

        out.append(enc_state)

        return list(reversed(out))

    def _get_obs(self):
        return {
            "field_state": self.field_state,
            "life_points": self.life_points,
            "weather_type": self.weather_type,
        }

    def step(self, action):
        self._get_new_day(with_life_points=False)

        enc_state = self.encode(sum(self.field_state), self.life_points, self.weather_type)
        P = self.P[enc_state][action]
        
        if action:
            chance = np.random.sample()
            chance = (chance - 0.1) if self.weather_type else chance
            
            if chance >= P[0][0]:
                p, new_state, reward, self.is_alive = P[1]
            else:
                p, new_state, reward, self.is_alive = P[0]
        else:
            chance = 0
            p, new_state, reward, self.is_alive = P[0]

        _, self.life_points, _ = self.decode(new_state)
        
        self.episode_length -= 1

        return self._get_obs(), reward, self.is_alive, {"chance": chance}

    def reset(self, seed=42):
        np.random.seed(42)

        self.episode_length = 6
        self._get_new_day(with_life_points=True)
        
        return self._get_obs()

    def render(self, mode="human", close=False):
        print(f"--Days left: {self.episode_length}")
        print(f"--State of Field: {self.field_state}")
        print(f"--Current life: {self.life_points}")
        print(f"--Type of Weather: {self.WEATHER_DICT[self.weather_type]}")