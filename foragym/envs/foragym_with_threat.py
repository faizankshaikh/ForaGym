import numpy as np
import pandas as pd

from gymnasium import spaces, Env
from itertools import product


class ForaGym_with_threat(Env):
    """
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, items_path=""):
        self.render_mode = render_mode

        self.action_dict = {0: "wait", 1: "forage"}
        consequences = [
            "Left environment / Forage failed / No threat encountered",
            "Right environment / Forage failed / No threat encountered",
            "Left environment / Forage successful / No threat encountered",
            "Right environment / Forage successful / No threat encountered",
            "Left environment / Forage failed / Threat encountered",
            "Right environment / Forage failed / Threat encountered",
            "Waited",
        ]
        self.consequences_dict = dict(zip(np.arange(0, 7), consequences))

        if not items_path:
            self.items_path = "foragym/data/items_with_threat.csv"
        else:
            self.items_path = items_path

        self.num_days = 9
        self.num_life_points = 7
        self.done = False
        self.env_choice = 0

        self.forests = self._get_forests(self.items_path)
        self.num_forests = len(self.forests)

        self.num_envs = 2

        self.nS = self.num_days * self.num_life_points * self.num_forests
        self.nA = len(self.action_dict)

        self._get_transition_matrix()

        self.observation_space = spaces.Dict(
            {
                "days_left": spaces.Discrete(self.num_days),
                "life_points_left": spaces.Discrete(self.num_life_points),
                "environment": spaces.Box(
                    low=0, high=2.0, shape=(3,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(self.nA)
        self._init_episode()

    def _get_forests(self, items_path):
        return pd.read_csv(items_path)

    def _get_consequences(self, payload):
        if payload["consequence_id"] == 0 and payload["action"] == 1:
            transition_prob = (1 - payload["forest_quality_left"]) * (
                1 - payload["threat_encounter_left"]
            )
        elif payload["consequence_id"] == 1 and payload["action"] == 1:
            transition_prob = (1 - payload["forest_quality_right"]) * (
                1 - payload["threat_encounter_right"]
            )
        elif payload["consequence_id"] == 2 and payload["action"] == 1:
            transition_prob = payload["forest_quality_left"] * (
                1 - payload["threat_encounter_left"]
            )
        elif payload["consequence_id"] == 3 and payload["action"] == 1:
            transition_prob = payload["forest_quality_right"] * (
                1 - payload["threat_encounter_right"]
            )
        elif payload["consequence_id"] == 4 and payload["action"] == 1:
            transition_prob = payload["threat_encounter_left"]
        elif payload["consequence_id"] == 5 and payload["action"] == 1:
            transition_prob = payload["threat_encounter_right"]
        elif payload["consequence_id"] == 6 and payload["action"] == 0:
            transition_prob = 2
        else:
            return []

        transition_prob /= self.num_envs

        new_days_left = payload["days_left"] - 1

        if payload["consequence_id"] in (0, 1) and payload["action"] == 1:
            new_life_points_left = payload["life_points_left"] - 2
        elif payload["consequence_id"] == 2 and payload["action"] == 1:
            new_life_points_left = (
                payload["life_points_left"] + payload["nutritional_quality_left"]
            )
        elif payload["consequence_id"] == 3 and payload["action"] == 1:
            new_life_points_left = (
                payload["life_points_left"] + payload["nutritional_quality_right"]
            )
        elif payload["consequence_id"] in (4, 5) and payload["action"] == 1:
            new_life_points_left = payload["life_points_left"] - 3
        elif payload["consequence_id"] == 6 and payload["action"] == 0:
            new_life_points_left = payload["life_points_left"] - 1
        else:
            return []

        new_life_points_left = np.clip(
            new_life_points_left, 0, self.num_life_points - 1
        )

        new_enc_state = self.encode(
            new_days_left, new_life_points_left, payload["forest_type"]
        )

        if not new_life_points_left:
            reward = -1
            done = True
        elif not new_days_left:
            reward = 0
            done = True
        else:
            reward = 0
            done = False

        return [transition_prob, new_enc_state, reward, done]

    def _get_transition_matrix(self):
        self.P = {
            state: {action: [] for action in range(self.nA)} for state in range(self.nS)
        }

        for days_left in range(1, self.num_days):
            for life_points_left in range(1, self.num_life_points):
                for forest_type in range(self.num_forests):
                    enc_state = self.encode(days_left, life_points_left, forest_type)

                    forest_quality_left = self.forests.loc[
                        self.forests["forest_type"] == forest_type,
                        "forest_quality_left",
                    ].values[0]
                    threat_encounter_left = self.forests.loc[
                        self.forests["forest_type"] == forest_type,
                        "threat_encounter_left",
                    ].values[0]
                    nutritional_quality_left = self.forests.loc[
                        self.forests["forest_type"] == forest_type,
                        "nutritional_quality_left",
                    ].values[0]
                    forest_quality_right = self.forests.loc[
                        self.forests["forest_type"] == forest_type,
                        "forest_quality_right",
                    ].values[0]
                    threat_encounter_right = self.forests.loc[
                        self.forests["forest_type"] == forest_type,
                        "threat_encounter_right",
                    ].values[0]
                    nutritional_quality_right = self.forests.loc[
                        self.forests["forest_type"] == forest_type,
                        "nutritional_quality_right",
                    ].values[0]

                    for action in range(self.nA):
                        for consequence_id in range(0, len(self.consequences_dict)):
                            payload = {}
                            payload["days_left"] = days_left
                            payload["life_points_left"] = life_points_left
                            payload["forest_type"] = forest_type
                            payload["forest_quality_left"] = forest_quality_left
                            payload["threat_encounter_left"] = threat_encounter_left
                            payload[
                                "nutritional_quality_left"
                            ] = nutritional_quality_left
                            payload["forest_quality_right"] = forest_quality_right
                            payload["threat_encounter_right"] = threat_encounter_right
                            payload[
                                "nutritional_quality_right"
                            ] = nutritional_quality_right
                            payload["action"] = action
                            payload["consequence_id"] = consequence_id

                            payload["result"] = self._get_consequences(payload)

                            if payload["result"]:
                                self.P[enc_state][action].append(payload["result"])

    def _init_episode(self):
        self.days_left = self.num_days - 1
        
        self.life_points_left = np.random.randint(4, 6)
        self.forest_type = np.random.randint(0, self.num_forests - 1)

        self.forest_quality_left = self.forests.loc[
            self.forests["forest_type"] == self.forest_type, "forest_quality_left"
        ].values[0]
        self.threat_encounter_left = self.forests.loc[
            self.forests["forest_type"] == self.forest_type, "threat_encounter_left"
        ].values[0]
        self.nutritional_quality_left = self.forests.loc[
            self.forests["forest_type"] == self.forest_type, "nutritional_quality_left"
        ].values[0]
        self.forest_quality_right = self.forests.loc[
            self.forests["forest_type"] == self.forest_type, "forest_quality_right"
        ].values[0]
        self.threat_encounter_right = self.forests.loc[
            self.forests["forest_type"] == self.forest_type, "threat_encounter_right"
        ].values[0]
        self.nutritional_quality_right = self.forests.loc[
            self.forests["forest_type"] == self.forest_type, "nutritional_quality_right"
        ].values[0]

    def encode(self, days_left, life_points_left, forest_type):
        enc_state = days_left

        enc_state *= self.num_life_points
        enc_state += life_points_left

        enc_state *= self.num_forests
        enc_state += forest_type

        return enc_state

    def decode(self, enc_state):
        out = []
        out.append(enc_state % self.num_forests)
        enc_state = enc_state // self.num_forests

        out.append(enc_state % self.num_life_points)
        enc_state = enc_state // self.num_life_points

        out.append(enc_state)

        return list(reversed(out))

    def _get_obs(self):

        return {
            "days_left": int(self.days_left),
            "life_points_left": int(self.life_points_left),
            "environment": np.array(
                [self.forest_quality, self.threat_encounter, self.nutritional_quality],
                dtype=np.float32,
            ),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._init_episode()

        self.env_choice = np.random.choice([0, 1])

        if self.env_choice:
            self.forest_quality = self.forest_quality_left
            self.threat_encounter = self.threat_encounter_left
            self.nutritional_quality = self.nutritional_quality_left
        else:
            self.forest_quality = self.forest_quality_right
            self.threat_encounter = self.threat_encounter_right
            self.nutritional_quality = self.nutritional_quality_right

        if self.render_mode == "human":
            self.render_text(is_start=True)

        return self._get_obs(), {"env_choice": self.env_choice}

    def render(self):
        if self.render_mode == "human":
            self.render_text()

    def render_text(self, is_start=False):
        if is_start:
            print(
                f"--Forest Quality for the left environment: {self.forest_quality_left:.2f}"
            )
            print(
                f"--Threat Encounter probability for the left environment: {self.threat_encounter_left:.2f}"
            )
            print(
                f"--Nutritional Quality for the left environment: {self.nutritional_quality_left:.2f}"
            )
            print(
                f"--Forest Quality for the right environment: {self.forest_quality_right:.2f}"
            )
            print(
                f"--Threat Encounter probability for the right environment: {self.threat_encounter_right:.2f}"
            )
            print(
                f"--Nutritional Quality for the right environment: {self.nutritional_quality_right:.2f}"
            )
            print("-" * 10)
            print("")
        else:
            print(f"--Consequence: {self.consequences_dict[self.consequence_id]}")
            print(f"--Reward?: {self.reward}")
            print(f"--Episode done?: {self.done}")

        print(f"--Days left: {self.days_left}")
        print(f"--Current life: {self.life_points_left}")
        print(f"--Current Forest Quality: {self.forest_quality:.2f}")
        print(f"--Current Threat Encounter probability: {self.threat_encounter:.2f}")
        print(f"--Current Nutritional Quality: {self.nutritional_quality:.2f}")

    def close(self):
        pass

    def step(self, action):
        if self.days_left <= 0:
            self.done = True
            return self._get_obs(), self.reward, self.done, False, {"env_choice": self.env_choice}

        enc_state = self.encode(self.days_left, self.life_points_left, self.forest_type)
        P = self.P[enc_state][action]

        if action:
            probs = [prob for (prob, _, _, _) in P]
            if self.env_choice:
                probs = [i * 2 for i in probs[::2]]
            else:
                probs = [i * 2 for i in probs[1::2]]
            self.consequence_id = np.random.choice(np.arange(0, 3), p=probs)
            if self.env_choice:
                self.consequence_id *= 2
            else:
                self.consequence_id = self.consequence_id * 2 + 1
            prob, new_enc_state, self.reward, self.done = P[self.consequence_id]
        else:
            self.consequence_id = 6
            prob, new_enc_state, self.reward, self.done = P[0]

        self.env_choice = np.random.choice([0, 1])
        if self.env_choice:
            self.forest_quality = self.forest_quality_left
            self.threat_encounter = self.threat_encounter_left
            self.nutritional_quality = self.nutritional_quality_left
        else:
            self.forest_quality = self.forest_quality_right
            self.threat_encounter = self.threat_encounter_right
            self.nutritional_quality = self.nutritional_quality_right

        self.days_left, self.life_points_left, _ = self.decode(new_enc_state)

        if self.render_mode == "human":
            self.render_text(is_start=False)

        return self._get_obs(), float(self.reward), self.done, False, {"env_choice": self.env_choice}
