from gym_env.env import Action, Stage
import numpy as np
import pandas as pd
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
        epsilon,
        alpha,
        gamma,
        name,
        stack_size,
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

        # state-action history
        self.history = []

        # Track if we've updated for the current hand
        self.last_update_hand = -1
        self.current_hand = 0

        # Training metrics
        self.action_rewards = []  # Track reward per action

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

        print(f"[Agent] Initialized weights: {self.weights.shape}")

    # =====================================================================
    # CORE METHODS
    # =====================================================================

    def action(self, legal_actions: List[Action], observation: np.ndarray, info: Dict, funds_history: pd.DataFrame) -> Action:

        _ = observation  # not used

        stage = info["community_data"]["stage"]
        my_position = info["player_data"]["position"]

        # Detect if we're at the start of a new hand by checking:
        # 1. We're in preflop (stage[0] == True)
        # 2. The funds_history has grown (new hand was recorded)
        # 3. We haven't updated for this hand yet
        is_new_hand = (
            stage[0] and  # In preflop
            len(funds_history) > self.current_hand and  # New hand recorded in funds_history
            self.last_update_hand < self.current_hand and  # Haven't updated yet
            len(self.history) > 0  # Have history to update from
        )

        # Update weights from previous hand when starting a new hand
        if is_new_hand:
            # print(f"!! Call Update (Hand {self.current_hand} -> {len(funds_history)}) !!")
            self._update(funds_history, my_position)
            self.history = [] # clear history for new hand
            self.last_update_hand = self.current_hand

        # Update current hand counter
        if len(funds_history) > self.current_hand:
            self.current_hand = len(funds_history)

        action = None
        if not legal_actions:
            action = Action.FOLD
        elif np.random.random() < self.epsilon:
            action = np.random.choice(legal_actions)
        else:
            action = self._choose_greedy(legal_actions, info)
        
        # Save the state of each stage and the agent's actions for weight training
        # print(f"\nHand: {self.current_hand}, Stage: {np.argmax(stage)}, Action: {action}\n")
        # print("Legal actions:", legal_actions)

        # to avoid reference bugs
        import copy
        saved_info = copy.deepcopy(info)

        self.history.append((saved_info, action))
        return action


    def _choose_greedy(self, legal_actions: List[Action], info: Dict) -> Action:
        features = self._extract_features(info)
        q_values = np.dot(features, self.weights)
        legal_q = [(a, q_values[self.action_order.index(a)]) for a in legal_actions]
        return max(legal_q, key=lambda x: x[1])[0]
    

    def _update(self, funds_history: pd.DataFrame, my_position: int):
        if len(self.history) == 0:
            print("History is empty")
            return

        # Loop through each action in the hand
        for i, (info, action) in enumerate(self.history):

            # Calculate reward and get next state info or None if last action/state in hand

            stack_before_action = self.history[i][0]['player_data']['stack'][my_position]
            
            stack_after_action = 0
            next_info = None
            if i + 1 < len(self.history):
                # get agents stack in action+1 index of history (after action taken)
                stack_after_action = self.history[i + 1][0]['player_data']['stack'][my_position]
                # get next state info from next action index in history
                next_info = self.history[i + 1][0] if i + 1 < len(self.history) else None
            else:
                # if last action/state of hand, then get agents stack at end of hand from funds_history
                stack_after_action = funds_history.iloc[self.current_hand - 1][my_position]

            reward = stack_after_action - stack_before_action
            self.action_rewards.append(reward)

            self._q_learning_update(info, action, reward, next_info)


    def _q_learning_update(self,
                           info: Dict,
                           action: Action,
                           reward: float,
                           next_info: Dict):
        
        # may want to do something to incorporate the initial stack size into the reward calculation
        
        features = self._extract_features(info) 
        q_values = np.dot(features, self.weights)
        a_idx = self.action_order.index(action)
        q_sa = q_values[a_idx]

        if next_info is not None:
            next_features = self._extract_features(next_info)
            next_q = np.dot(next_features, self.weights)
            target = reward + self.gamma * np.max(next_q)
        else:
            target = reward
        
        td_error = target - q_sa
        self.weights[:, a_idx] += self.alpha * td_error * features


    # =====================================================================
    # FEATURES
    # =====================================================================

    def _extract_features(self, info: Dict) -> np.ndarray:
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
            print(f"[Agent] Loaded weights: {filename}")
        else:
            raise FileNotFoundError(filename)


    # =====================================================================
    # TRAINING METRICS AND VISUALIZATION
    # ====================================================================
    def plot_training_progress(self):
        """Plot training metrics to visualize learning"""
        import matplotlib.pyplot as plt
        
        if len(self.action_rewards) == 0:
            print("No action rewards data to plot yet.")
            return
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.action_rewards, label='Action Rewards', color='blue', alpha=0.6)
        plt.xlabel('Action Number')
        plt.ylabel('Reward')
        plt.title('Training Progress: Action Rewards Over Time')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print(f"\n[Agent] Training plot saved as 'training_progress.png'")
        plt.show()

    def __str__(self):
        return self.name