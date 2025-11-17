from gym_env.env import Action
import numpy as np
import os
from typing import List, Dict, Any, Tuple


# Configuration
FEATURE_SIZE = 13
NUM_ACTIONS = 8


class Player:
    """
    Final Linear Q-Learning Poker Agent with stable training.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 0.005,
        gamma: float = 0.95,
        name: str = "MyAgent",
        stack_size: int = None,
        weights_file: str = None
    ):
        # Required by environment
        self.name = name
        self.autoplay = True
        self.initial_stack = stack_size

        # Hyperparameters
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training_error = []
        self.steps = 0

        self.last_exploiting = False
        self.last_obs = {}

        # Action mapping
        self.action_order = [
            Action.FOLD,
            Action.CHECK,
            Action.CALL,
            Action.RAISE_3BB,
            Action.RAISE_HALF_POT,
            Action.RAISE_POT,
            Action.RAISE_2POT,
            Action.ALL_IN
        ]

        # Weights
        if weights_file and os.path.exists(weights_file):
            self.load_weights(weights_file)
        else:
            self.weights = np.random.uniform(-0.1, 0.1, (FEATURE_SIZE, NUM_ACTIONS))
            self.target_weights = self.weights.copy()

        print(f"[Agent] Initialized weights: {self.weights.shape}")

    # =====================================================================
    # CORE METHODS
    # =====================================================================

    def action(self, legal_actions: List[Action], observation: np.ndarray, info: Dict) -> Action:
        if not legal_actions:
            return Action.FOLD

        if np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)
        else:
            return self._choose_greedy(observation, legal_actions, info)


    def _choose_greedy(self, obs_array: np.ndarray, legal_actions: List[Action], info: Dict) -> Action:
        features = self.extract_features(obs_array, info)
        q_values = np.dot(features, self.weights)
        legal_q = [(a, q_values[self.action_order.index(a)]) for a in legal_actions]
        return max(legal_q, key=lambda x: x[1])[0]
    

    def update(self, obs: np.ndarray, info: dict, action: Action, reward: float, 
           next_obs: np.ndarray, next_info: dict, terminated: bool):
        
        self.steps += 1

        # Use the info dict
        features = self.extract_features(obs, info) 

        q_values = np.dot(features, self.weights)
        a_idx = self.action_order.index(action)
        q_sa = q_values[a_idx]

        if terminated:
            target = reward
        else:
            # Use the next_info dict
            next_features = self.extract_features(next_obs, next_info)
            next_q = np.dot(next_features, self.target_weights)
            print("!!Reward used in update: ", reward)
            target = reward + self.gamma * np.max(next_q)

        td_error = target - q_sa
        self.training_error.append(td_error)
        self.weights[:, a_idx] += self.alpha * td_error * features
        
    # =====================================================================
    # FEATURES
    # =====================================================================

    def extract_features(self, obs_array: np.ndarray, info: Dict) -> np.ndarray:
        """
        Extract binary features from observation.

        Features (13 total):
        0. equity >= 0.5 (strong hand)
        1. equity >= 0.7 (very strong hand)
        2. stack >= 0.5 (comfortable stack)
        3. stack >= 1.0 (full or above stack)
        4. stage == preflop (0)
        5. stage == flop (1)
        6. stage == turn (2)
        7. stage == river (3)
        8. community_pot >= 0.5  -the total pot for the hand
        9. community_pot >= 1.0
        10. current_stage_pot >= 0.25  -the pot for the current betting round
        11. current_stage_pot >= 0.5
        12. current_stage_pot >= 1.0
        """
        features = np.zeros(FEATURE_SIZE, dtype=np.float64)

        # Extract from info dict
        player_data = info['player_data']
        community_data = info['community_data']

        # Equity bins
        equity = player_data['equity_to_river_alive']
        equity = 0.0 if np.isnan(equity) else float(equity)
        features[0] = 1.0 if equity >= 0.5 else 0.0
        features[1] = 1.0 if equity >= 0.7 else 0.0

        # Stack bins
        my_position = player_data['position']
        my_stack = player_data['stack'][my_position]
        features[2] = 1.0 if my_stack >= 0.5 else 0.0
        features[3] = 1.0 if my_stack >= 1.0 else 0.0

        # Stage one-hot encoding
        stage_one_hot = community_data['stage']
        stage_idx = int(np.argmax(stage_one_hot))
        features[4] = 1.0 if stage_idx == 0 else 0.0  # preflop
        features[5] = 1.0 if stage_idx == 1 else 0.0  # flop
        features[6] = 1.0 if stage_idx == 2 else 0.0  # turn
        features[7] = 1.0 if stage_idx == 3 else 0.0  # river

        # Community pot bins
        c_pot = float(community_data['community_pot'])
        features[8] = 1.0 if c_pot >= 0.5 else 0.0
        features[9] = 1.0 if c_pot >= 1.0 else 0.0

        # Current stage pot bins
        r_pot = float(community_data['current_round_pot'])
        features[10] = 1.0 if r_pot >= 0.25 else 0.0
        features[11] = 1.0 if r_pot >= 0.5 else 0.0
        features[12] = 1.0 if r_pot >= 1.0 else 0.0

        return features

    # =====================================================================
    # PERSISTENCE
    # =====================================================================

    def save_weights(self, filename: str):
        np.save(filename, self.weights)
        print(f"[Agent] Saved weights: {filename}")

    def load_weights(self, filename: str):
        if os.path.exists(filename):
            self.weights = np.load(filename)
            if self.weights.shape != (FEATURE_SIZE, NUM_ACTIONS):
                raise ValueError(f"Shape mismatch: {self.weights.shape}")
            self.target_weights = self.weights.copy()
            print(f"[Agent] Loaded weights: {filename}")
        else:
            raise FileNotFoundError(filename)

    def __str__(self):
        return self.name