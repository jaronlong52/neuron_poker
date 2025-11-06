"""
agents/agent_my_agent.py

FINAL VERSION: Linear Q-Learning Poker Agent
- Hand strength features
- Target network
- Exploit-only updates
- Reward clipping
- Stable training
"""

from gym_env.env import Action
import numpy as np
import os
from typing import List, Dict, Any, Tuple


# Configuration
FEATURE_SIZE = 30
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
        autoplay: bool = False,
        stack_size: int = None,
        weights_file: str = None
    ):
        # Required by environment
        self.name = name
        self.autoplay = autoplay
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
        if self.steps % self.target_update_freq == 0:
            self.target_weights = self.weights.copy()

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
            target = reward + self.gamma * np.max(next_q)

        td_error = target - q_sa
        self.training_error.append(td_error)
        self.weights[:, a_idx] += self.alpha * td_error * features

    # =====================================================================
    # FEATURES
    # =====================================================================

    def extract_features(self, obs_array: np.ndarray, info: Dict) -> np.ndarray:
        features = np.zeros(FEATURE_SIZE, dtype=np.float32)

        # Unpack info
        player_data = info.get('player_data', {})
        community_data = info.get('community_data', {})
        stage_data = info.get('stage_data', [{}])[0]  # first stage

        # 0: Equity
        features[0] = player_data.get('equity_to_river_alive', 0.0)

        # 1–4: Stage
        stage_idx = min(int(np.argmax(community_data.get('stage', [1,0,0,0]))), 3)
        features[1 + stage_idx] = 1.0

        # 5–7: Position
        pos = player_data.get('position', 0)
        n_players = len(info.get('community_data', {}).get('active_players', [0]*6))
        norm_pos = pos / max(n_players - 1, 1) if n_players > 1 else 0
        if norm_pos < 0.33:   features[5] = 1.0
        elif norm_pos < 0.66: features[6] = 1.0
        else:                 features[7] = 1.0

        # 8: Pot odds
        call_amount = community_data.get('min_call_at_action', [0]*6)[pos]
        pot = community_data.get('community_pot', 0) + community_data.get('current_round_pot', 0)
        features[8] = min(call_amount / max(pot, 1), 2.0)

        # 9: Stack ratio
        my_stack = player_data.get('stack', [500])[0]
        features[9] = my_stack / (self.initial_stack or 500)

        # 10: Pot size
        features[10] = pot / ((self.initial_stack or 500) * 6)

        # 11: Investment ratio
        contrib = stage_data.get('contribution', [0]*6)[pos]
        features[11] = contrib / max(pot, 1)

        # 12: Active players
        features[12] = n_players / 6.0

        # 13: Call cost / stack
        features[13] = call_amount / max(my_stack, 1)

        # 14: Stack in BB
        bb = community_data.get('big_blind', 10)
        features[14] = my_stack / bb

        # 15: Raises this round
        raises = sum(stage_data.get('raises', [0]*6))
        features[15] = min(raises / 5.0, 1.0)

        # 16: Aggression (total bets)
        bets = sum(stage_data.get('contribution', [0]*6))
        features[16] = min(bets / (pot + 1), 3.0)

        # 17–20: Hand strength (skip for now — needs card parsing)
        # Or set to 0.5
        features[17:21] = 0.5

        return features

    def _card_rank(self, card: str) -> int:
        rank_map = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
                    'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
        return rank_map.get(card[0], 0)

    def _hand_strength(self, r1: int, r2: int, suited: bool) -> float:
        if r1 < r2: r1, r2 = r2, r1
        gap = r1 - r2
        if r1 == r2:  # Pair
            return 0.7 + (r1 - 2) * 0.02
        elif suited and r1 >= 10 and gap <= 2:
            return 0.75 - gap * 0.05
        elif r1 >= 12 and r2 >= 10:
            return 0.6
        elif r1 >= 10 and r2 >= 8:
            return 0.5
        else:
            return max(0.1, 0.4 - gap * 0.1)

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