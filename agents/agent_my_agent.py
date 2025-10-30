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
NUM_ACTIONS = 7


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
        weights_file: str = None,
        target_update_freq: int = 1000
    ):
        # Required by environment
        self.name = name
        self.stack = stack_size
        self.autoplay = False

        # Hyperparameters
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training_error = []
        self.target_update_freq = target_update_freq
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

    def action(self, action_space: List[Action], observation: Any, info: Dict) -> Action:
        obs_dict = info.get('observation', {})
        valid_actions = action_space
    
        if not valid_actions:
            return Action.FOLD
    
        exploiting = np.random.random() >= self.epsilon
        if exploiting:
            action = self._choose_greedy(obs_dict, valid_actions)
        else:
            action = np.random.choice(valid_actions)
    
        self.last_exploiting = exploiting
        self.last_obs = obs_dict
    
        return action

    def _choose_greedy(self, obs_dict: Dict, valid_actions: List[Action]) -> Action:
        q_values = np.dot(self.extract_features(obs_dict), self.weights)
        valid_q = []
        valid_act = []

        for a in valid_actions:
            if a in self.action_order:
                idx = self.action_order.index(a)
                valid_q.append(q_values[idx])
                valid_act.append(a)

        if not valid_act:
            return np.random.choice(valid_actions)

        return valid_act[int(np.argmax(valid_q))]

    # =====================================================================
    # FEATURES
    # =====================================================================

    def extract_features(self, obs_dict: Dict) -> np.ndarray:
        features = np.zeros(FEATURE_SIZE, dtype=np.float32)
        if not isinstance(obs_dict, dict):
            return features

        # 0: Equity
        eq = obs_dict.get('equity', {})
        features[0] = float(eq.get('me', 0.0))

        # 1–4: Stage
        stage = obs_dict.get('stage', 'PREFLOP')
        if stage == 'PREFLOP': features[1] = 1.0
        elif stage == 'FLOP':    features[2] = 1.0
        elif stage == 'TURN':    features[3] = 1.0
        elif stage == 'RIVER':   features[4] = 1.0

        # 5–7: Position
        pos = obs_dict.get('position', 0)
        n = max(obs_dict.get('active_players', 6), 1)
        norm = pos / max(n - 1, 1)
        if norm < 0.33:   features[5] = 1.0
        elif norm < 0.66: features[6] = 1.0
        else:             features[7] = 1.0

        # 8: Pot odds
        call = obs_dict.get('current_bet', 0) - obs_dict.get('my_contribution', 0)
        pot = obs_dict.get('pot', 1)
        features[8] = min(call / max(pot, 1), 2.0)

        # 9: Stack ratio
        features[9] = obs_dict.get('my_stack', 500) / (self.stack or 500)

        # 10: Pot size
        features[10] = pot / ((self.stack or 500) * 6)

        # 11: Investment ratio
        features[11] = obs_dict.get('my_contribution', 0) / max(pot, 1)

        # 12: Active players
        features[12] = n / 6.0

        # 13: Call cost / stack
        features[13] = call / max(obs_dict.get('my_stack', 1), 1)

        # 14: Stack in BB
        bb = obs_dict.get('big_blind', 10)
        features[14] = obs_dict.get('my_stack', 500) / bb

        # 15: Raises this round
        features[15] = min(obs_dict.get('num_raises_this_round', 0) / 5.0, 1.0)

        # 16: Aggression
        bets = sum(obs_dict.get('bets_this_round', [0]))
        features[16] = min(bets / (pot + 1), 3.0)

        # 17–20: Hand strength
        hand = obs_dict.get('hole_cards', [])
        if len(hand) == 2:
            r1 = self._card_rank(hand[0])
            r2 = self._card_rank(hand[1])
            suited = hand[0][-1] == hand[1][-1]
            strength = self._hand_strength(r1, r2, suited)
            features[17] = strength
            features[18] = 1.0 if strength > 0.8 else 0.0
            features[19] = 1.0 if strength > 0.6 else 0.0
            features[20] = 1.0 if strength > 0.4 else 0.0

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
    # UPDATE
    # =====================================================================

    def update(self, obs: Dict, action: Action, reward: float, next_obs: Dict, terminated: bool):
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_weights = self.weights.copy()

        features = self.extract_features(obs)
        q_values = np.dot(features, self.weights)

        if action not in self.action_order:
            return
        a_idx = self.action_order.index(action)
        q_sa = q_values[a_idx]

        if terminated:
            target = reward
        else:
            next_features = self.extract_features(next_obs)
            next_q = np.dot(next_features, self.target_weights)
            target = reward + self.gamma * np.max(next_q)

        td_error = target - q_sa
        self.training_error.append(td_error)
        self.weights[:, a_idx] += self.alpha * td_error * features

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