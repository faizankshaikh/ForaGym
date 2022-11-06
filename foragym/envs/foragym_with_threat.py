import gym
import numpy as np

from gym import spaces
from itertools import product


class ForaGym_with_threat(gym.Env):
    """
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.action_dict = {0: "wait", 1: "forage"}
        consequences = [
            "Left environment / Forage failed / No threat encountered",
            "Right environment / Forage failed / No threat encountered",
            "Left environment / Forage successful / No threat encountered",
            "Right environment / Forage successful / No threat encountered",
            "Left environment / Forage failed / Threat encountered",
            "Right environment / Forage failed / Threat encountered",
            "Waited"
        ]
        self.consequence_dict = dict(zip(np.arange(0, 7), consequences))

        self.num_days_left = 8
        self.num_life_points_left = 6
        self.done = False

        self.forest_quality = np.arange(0.5, 0.8, 0.1)
        self.forest_qualities_list = [(i,j) for i, j in product(self.forest_quality, self.forest_quality) if i < j]
        self.num_forest_qualities = len(self.forest_qualities_list)

        self.threat_encounter = np.arange(0.1, 0.4, 0.1)
        self.threat_encounter_list = [(i,j) for i, j in product(self.threat_encounter, self.threat_encounter) if i != j]
        self.num_threat_encounter = len(self.threat_encounter_list)

        self.forest = [((i, k), (j, l)) for ((i, j), (k, l)) in product(self.forest_qualities_list, self.threat_encounter_list)]
        self.num_forest = len(self.forest)

        self.num_envs = 2

        self.nS = (
            self.num_days_left * self.num_life_points_left * self.num_forest
        )
        self.nA = len(self.action_dict)

        self._get_transition_matrix()

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Dict(
            {
                "days_left": spaces.Discrete(self.num_days_left),
                "life_points_left": spaces.Discrete(self.num_life_points_left),
                "forest_type": spaces.Discrete(self.num_forest)
            }
        )

        self._init_episode()

    def _get_transition_matrix(self):
        self.P = {
            state: {action: [] for action in range(self.nA)}
            for state in range(self.nS)
        }

        for days_left in range(1, self.num_days_left):
            for life_points_left in range(1, self.num_life_points_left):
                for forest_type in range(self.num_forest):
                    enc_state = self.encode(days_left, life_points_left, forest_type)

                    (forest_quality_left, threat_encounter_left), (forest_quality_right, threat_encounter_right) = self.forest[forest_type]

                    for action in range(self.nA):
                        if action:
                            # forage, no threat but fail (left)
                            new_days_left = np.clip(days_left - 1, 0, self.num_days_left)
                            new_life_points_left = np.clip(life_points_left - 2, 0, self.num_life_points_left - 1)
                            new_forest_type = forest_type

                            prob =  np.clip((1 - forest_quality_left)*(1 - threat_encounter_left), 0, 1) / 2
                            new_enc_state = self.encode(new_days_left, new_life_points_left, new_forest_type)
                            if not new_life_points_left:
                                reward = -1
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            elif not new_days_left:
                                reward = 0
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            else:
                                reward = 0
                                done = False
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])

                            # forage, no threat but fail (right)
                            new_days_left = np.clip(days_left - 1, 0, self.num_days_left)
                            new_life_points_left = np.clip(life_points_left - 2, 0, self.num_life_points_left - 1)
                            new_forest_type = forest_type

                            prob =  np.clip((1 - forest_quality_right)*(1 - threat_encounter_right), 0, 1) / 2
                            new_enc_state = self.encode(new_days_left, new_life_points_left, new_forest_type)
                            if not new_life_points_left:
                                reward = -1
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            elif not new_days_left:
                                reward = 0
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            else:
                                reward = 0
                                done = False
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            

                            # forage, no threat and succeed (left)
                            new_days_left = np.clip(days_left - 1, 0, self.num_days_left)
                            new_life_points_left = np.clip(life_points_left + 1, 0, self.num_life_points_left - 1)
                            new_forest_type = forest_type

                            prob = np.clip(((forest_quality_left)*(1 - threat_encounter_left)), 0, 1) / 2
                            new_enc_state = self.encode(new_days_left, new_life_points_left, new_forest_type)
                            if not new_life_points_left:
                                reward = -1
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            elif not new_days_left:
                                reward = 0
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            else:
                                reward = 0
                                done = False
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])

                            # forage, no threat and succeed (right)
                            new_days_left = np.clip(days_left - 1, 0, self.num_days_left)
                            new_life_points_left = np.clip(life_points_left + 1, 0, self.num_life_points_left - 1)
                            new_forest_type = forest_type

                            prob = np.clip(((forest_quality_right)*(1 - threat_encounter_right)), 0, 1) / 2
                            new_enc_state = self.encode(new_days_left, new_life_points_left, new_forest_type)
                            if not new_life_points_left:
                                reward = -1
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            elif not new_days_left:
                                reward = 0
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            else:
                                reward = 0
                                done = False
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])

                            # forage but threat found (left)
                            new_days_left = np.clip(days_left - 1, 0, self.num_days_left)
                            new_life_points_left = np.clip(life_points_left - 3, 0, self.num_life_points_left - 1)
                            new_forest_type = forest_type

                            prob = threat_encounter_left / 2
                            new_enc_state = self.encode(new_days_left, new_life_points_left, new_forest_type)
                            if not new_life_points_left:
                                reward = -1
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            elif not new_days_left:
                                reward = 0
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            else:
                                reward = 0
                                done = False
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])

                            # forage but threat found (right)
                            new_days_left = np.clip(days_left - 1, 0, self.num_days_left)
                            new_life_points_left = np.clip(life_points_left - 3, 0, self.num_life_points_left - 1)
                            new_forest_type = forest_type

                            prob = threat_encounter_right / 2
                            new_enc_state = self.encode(new_days_left, new_life_points_left, new_forest_type)
                            if not new_life_points_left:
                                reward = -1
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            elif not new_days_left:
                                reward = 0
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            else:
                                reward = 0
                                done = False
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                        else:
                            # wait
                            new_days_left = np.clip(days_left - 1, 0, self.num_days_left)
                            new_life_points_left = np.clip(life_points_left - 1, 0, self.num_life_points_left - 1)
                            new_forest_type = forest_type

                            prob = 1
                            new_enc_state = self.encode(new_days_left, new_life_points_left, new_forest_type)
                            if not new_life_points_left:
                                reward = -1
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            elif not new_days_left:
                                reward = 0
                                done = True
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])
                            else:
                                reward = 0
                                done = False
                                self.P[enc_state][action].append([prob, new_enc_state, reward, done])

    def _init_episode(self):
        self.days_left = self.num_days_left - 1

        self.life_points_left = 0
        while self.life_points_left < 4:
            self.life_points_left = self.observation_space["life_points_left"].sample()

        self.forest_type = self.observation_space["forest_type"].sample()

    def encode(self, days_left, life_points_left, forest_type):
        enc_state = days_left

        enc_state *= self.num_life_points_left
        enc_state += life_points_left

        enc_state *= self.num_forest
        enc_state += forest_type

        return enc_state

    def decode(self, enc_state):
        out = []
        out.append(enc_state % self.num_forest)
        enc_state = enc_state // self.num_forest

        out.append(enc_state % self.num_life_points_left)
        enc_state = enc_state // self.num_life_points_left

        out.append(enc_state)

        return list(reversed(out))

    def _get_obs(self):
        (forest_quality_left, threat_encounter_left), (forest_quality_right, threat_encounter_right) = self.forest[self.forest_type]


        return {
            "days_left": self.days_left,
            "life_points": self.life_points_left,
            "forest_quality_left": forest_quality_left,
            "forest_quality_right": forest_quality_right,
            "threat_encounter_left": threat_encounter_left,
            "threat_encounter_right": threat_encounter_right
        }

    def step(self, action):
        if self.days_left <= 0:
            self.done = True
            return self._get_obs(), reward, self.done, {"consequence": self.consequence_dict[consequence_id]}

        enc_state = self.encode(self.days_left, self.life_points_left, self.forest_type)
        P = self.P[enc_state][action]
        
        if action:
            probs = [prob for (prob, _, _, _) in P]
            consequence_id = np.random.choice(np.arange(0, 6), p=probs)
            prob, new_enc_state, reward, self.done = P[consequence_id]
        else:
            consequence_id = 0
            prob, new_enc_state, reward, self.done = P[0]

        self.days_left, self.life_points_left, _ = self.decode(new_enc_state)

        if self.render_mode == "human":
            self._render_text()

        return self._get_obs(), reward, self.done, {"consequence": self.consequence_dict[consequence_id]}

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)

        self._init_episode()

        if self.render_mode == "human":
            self._render_text()

        obs = self._get_obs()
        info = {}
        
        return obs, info

    def render(self):
        if self.render_mode == "human":
            self._render_text()

    def _render_text(self):
        (forest_quality_left, threat_encounter_left), (forest_quality_right, threat_encounter_right) = self.forest[self.forest_type]


        print(f"--Days left: {self.days_left}")
        print(f"--Current life: {self.life_points_left}")
        print(f"--Forest Quality for the left environment: {forest_quality_left:.2f}")
        print(f"--Forest Quality for the right environment: {forest_quality_right:.2f}")
        print(f"--Threat Encounter probability for the left environment: {threat_encounter_left:.2f}")
        print(f"--Threat Encounter probability for the right environment: {threat_encounter_right:.2f}")

    def close(self):
        pass