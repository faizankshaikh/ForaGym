# import important libraries and modules
import gym
import numpy as np

from itertools import product
from gym import spaces


class ForaGym(gym.Env):
    """Explanation of Foragym

    ### Description
    ### Actions
    ### Observations
    ### Reward
    ### Arguments
    """
    metadata = {"render_modes": ["human", "text"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.ACTION_DICT = {0: "Wait", 1: "Forage"}
        self.WEATHER_DICT = {0: "Clear", 1: "Rainy"}

        self.NUM_DAYS_LEFT = 6
        self.NUM_LIFE_POINTS = 7
        self.NUM_FIELDS = 5
        self.NUM_WEATHER_TYPES = 2
        self.BAD_WEATHER_EFFECT = 0.1
        self.NUM_ACTIONS = len(self.ACTION_DICT)

        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "days_left": spaces.Discrete(self.NUM_DAYS_LEFT),
                "life_points": spaces.Discrete(self.NUM_LIFE_POINTS),
                "field_state": spaces.MultiBinary(self.NUM_FIELDS),
                "weather_type": spaces.Discrete(self.NUM_WEATHER_TYPES)
            }
        )

        self.NUM_STATES = (
            self.NUM_DAYS_LEFT * self.NUM_LIFE_POINTS * (self.NUM_FIELDS + 1) * self.NUM_WEATHER_TYPES
        )

        self.NUM_INTERNAL_STATES = (self.NUM_FIELDS + 1) * self.NUM_WEATHER_TYPES

        self.is_dead = False
        
        self.P = {
            state: {action: [] for action in range(self.NUM_ACTIONS)}
            for state in range(self.NUM_STATES)
        }

        self._get_new_day(with_life_points=True)

        self._get_transition_probs()

    def _get_new_day(self, with_days_left=False, with_life_points=True):
        if with_days_left:
            self.days_left = 5


        if with_life_points:
            self.life_points = 0
            while not self.life_points:
                self.life_points = self.observation_space["life_points"].sample()

        field_state_sum = 0
        while not field_state_sum:
            self.field_state = self.observation_space["field_state"].sample()
            field_state_sum = sum(self.field_state)

        self.weather_type = self.observation_space["weather_type"].sample()



    def _get_transition_probs(self):
        for days_left in range(1, self.NUM_DAYS_LEFT):
            for life_point in range(1, self.NUM_LIFE_POINTS):
                for field in range(self.NUM_FIELDS + 1):
                    for weather in range(self.NUM_WEATHER_TYPES):
                        prob_success = np.clip(field / self.NUM_FIELDS - weather*self.BAD_WEATHER_EFFECT, 0, 1)
                        prob_failure = np.clip(1 - prob_success, 0, 1)

                        enc_state = self.encode(days_left, life_point, field, weather)

                        for action in range(self.NUM_ACTIONS):
                            if action:
                                # forage but fail
                                new_days_left_failure = np.clip(days_left - 1, 0, self.NUM_DAYS_LEFT - 1)
                                new_life_point_failure = np.clip(life_point - 2, 0, self.NUM_LIFE_POINTS - 1)
                                new_field_failure = list(range(self.NUM_FIELDS + 1))
                                new_weather_failure = list(range(self.NUM_WEATHER_TYPES))
                                new_states = list(product([new_days_left_failure], [new_life_point_failure], new_field_failure, new_weather_failure))
                                for new_days_left, new_life_point, new_field, new_weather in new_states:
                                    if not new_life_point:
                                        self.P[enc_state][action].append([prob_failure / len(new_states), enc_state, -1, True])
                                    elif not new_days_left:
                                        self.P[enc_state][action].append([prob_failure / len(new_states), enc_state, 0, True])
                                    else:
                                        new_enc_state = self.encode(new_days_left, new_life_point, new_field, new_weather)
                                        self.P[enc_state][action].append([prob_failure / len(new_states), new_enc_state, 0, False])
                                # forage and found
                                new_days_left_success = np.clip(days_left - 1, 0, self.NUM_DAYS_LEFT - 1)
                                new_life_point_success = np.clip(life_point + 1, 0, self.NUM_LIFE_POINTS - 1)
                                new_field_success = list(range(self.NUM_FIELDS + 1))
                                new_weather_success = list(range(self.NUM_WEATHER_TYPES))
                                new_states = list(product([new_days_left_success], [new_life_point_success], new_field_success, new_weather_success))
                                for new_days_left, new_life_point, new_field, new_weather in new_states:
                                    if not new_life_point:
                                        self.P[enc_state][action].append([prob_success / len(new_states), enc_state, -1, True])
                                    elif not new_days_left:
                                        self.P[enc_state][action].append([prob_success / len(new_states), enc_state, 0, True])
                                    else:
                                        new_enc_state = self.encode(new_days_left, new_life_point, new_field, new_weather)
                                        self.P[enc_state][action].append([prob_success / len(new_states), new_enc_state, 0, False])
                            else:
                                # wait
                                new_days_left_wait = np.clip(days_left - 1, 0, self.NUM_DAYS_LEFT - 1)
                                new_life_point_wait = np.clip(life_point - 1, 0, self.NUM_LIFE_POINTS - 1)
                                new_field_wait = list(range(self.NUM_FIELDS + 1))
                                new_weather_wait = list(range(self.NUM_WEATHER_TYPES))
                                new_states = list(product([new_days_left_wait], [new_life_point_wait], new_field_wait, new_weather_wait))
                                for new_days_left, new_life_point, new_field, new_weather in new_states:
                                    if not new_life_point:
                                        self.P[enc_state][action].append([1.0 / len(new_states), enc_state, -1, True])
                                    elif not new_days_left:
                                        self.P[enc_state][action].append([1.0 / len(new_states), enc_state, 0, True])
                                    else:
                                        new_enc_state = self.encode(new_days_left, new_life_point, new_field, new_weather)
                                        self.P[enc_state][action].append([1.0 / len(new_states), new_enc_state, 0, False])

    def encode(self, days_left, life_point, field, weather):
        enc_state = days_left

        enc_state *= self.NUM_LIFE_POINTS
        enc_state += life_point

        enc_state *= (self.NUM_FIELDS + 1)
        enc_state += field

        enc_state *= self.NUM_WEATHER_TYPES
        enc_state += weather

        return enc_state

    def decode(self, enc_state):
        out = []
        out.append(enc_state % self.NUM_WEATHER_TYPES)
        enc_state = enc_state // self.NUM_WEATHER_TYPES

        out.append(enc_state % (self.NUM_FIELDS + 1))
        enc_state = enc_state // (self.NUM_FIELDS + 1)

        out.append(enc_state % self.NUM_LIFE_POINTS)
        enc_state = enc_state // self.NUM_LIFE_POINTS

        out.append(enc_state)

        return list(reversed(out))

    def _get_obs(self):
        return {
            "days_left": self.days_left,
            "life_points": self.life_points,
            "field_state": self.field_state,
            "weather_type": self.weather_type,
        }

    def step(self, action):
        self._get_new_day(with_life_points=False)

        enc_state = self.encode(self.days_left, self.life_points, sum(self.field_state), self.weather_type)
        P = self.P[enc_state][action]
        
        if action:
            chance = np.random.sample()
            
            if chance >= P[0][0]*(self.NUM_INTERNAL_STATES):
                prob_transition, new_state, reward, self.is_dead = P[self.NUM_INTERNAL_STATES]
            else:
                prob_transition, new_state, reward, self.is_dead = P[0]
        else:
            chance = 0
            prob_transition, new_state, reward, self.is_dead = P[0]

        self.days_left, self.life_points, _, _ = self.decode(new_state)

        if self.days_left <= 0:
            self.is_dead = True

        return self._get_obs(), reward, self.is_dead, {"chance": chance, "prob_transition": prob_transition}

    def reset(self, seed=42):
        self._get_new_day(with_days_left=True)
        
        return self._get_obs()

    def render(self, mode="human", close=False):
        print(f"--Days left: {self.days_left}")
        print(f"--State of Field: {self.field_state}")
        print(f"--Current life: {self.life_points}")
        print(f"--Type of Weather: {self.WEATHER_DICT[self.weather_type]}")