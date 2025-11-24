from gym_env.env import Action, Stage
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Tuple


# Configuration
FEATURE_SIZE = 16
NUM_ACTIONS = 8


class Player:
    """
    Final Linear Q-Learning Poker Agent with stable training.
    """

    def __init__(
        self,
        epsilon,
        epsilon_decay,
        alpha,
        gamma,
        big_blind,
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
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.big_blind = big_blind

        # state-action history
        self.history = []

        # Track if we've updated for the current hand
        self.last_update_hand = -1
        self.current_hand = 0

        # Training metrics
        self.hand_rewards = []  # Track reward per action

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
        
        # print action chosen out of legal actions
        print(f"[Agent]  Chosen: {action.name},      Legal: {[a.name for a in legal_actions]}, Epsilon: {self.epsilon:.4f}")

        # to avoid reference bugs
        import copy
        saved_info = copy.deepcopy(info)

        self.history.append((saved_info, action))

        return action
    

    def add_history_marker(self):
        """Add a marker to indicate the end of a hand in history."""
        self.history.append((None, None))


    def _choose_greedy(self, legal_actions: List[Action], info: Dict) -> Action:
        features = self._extract_features(info)
        q_values = np.dot(features, self.weights)
        legal_q = [(a, q_values[self.action_order.index(a)]) for a in legal_actions]
        return max(legal_q, key=lambda x: x[1])[0]
    

    def update(self, funds_history: pd.DataFrame, my_position: int):
        if len(self.history) == 0:
            print("History is empty")
            return
        
        hand_count = 0

        # Loop through each action in the episode
        for i, (info, action) in enumerate(self.history):

            if info is None and action is None:
                print("Skipping marker")
                continue # skip hand start markers
        
            if i + 1 == len(self.history):
                print("End of history reached")
                break # end of history

            next_info = None 

            if self.history[i + 1][0] is None: # last action in hand
                stack_before_action = funds_history.iloc[hand_count].iloc[my_position]
                if hand_count + 1 >= len(funds_history):
                    print("No more funds history")
                    stack_after_action = stack_before_action
                    break
                else:
                    stack_after_action = funds_history.iloc[hand_count + 1].iloc[my_position]
                reward = stack_after_action - stack_before_action
                self.hand_rewards.append(reward)
                hand_count += 1
            else: # intermediate action
                reward = 0
                next_info = self.history[i + 1][0]

            self._q_learning_update(info, action, reward, next_info)
            print(f"[Agent] Update {i}: Action={action}, Reward={reward}")


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
        Extract features from observation.

        Features (16 total):
        0-1. Equity (continuous 0-1, then squared for non-linearity)
        2-3. Stack size (continuous normalized, then ratio to big blind)
        4-7. Stage (one-hot: preflop, flop, turn, river)
        8-9. Community pot (continuous normalized, then log scale)
        10-12. Current round pot (continuous normalized, then log, then sqrt)
        13. Pot odds (community_pot / (my_stack + current_bets))
        14. Opponent aggression (rough heuristic)
        15. Bias term (always 1.0)
        """
        features = np.zeros(FEATURE_SIZE, dtype=np.float64)

        # Extract from info dict
        player_data = info['player_data']
        community_data = info['community_data']

        # ========== EQUITY (continuous, not binary) ==========
        equity = player_data['equity_to_river_alive']
        equity = 0.0 if np.isnan(equity) else float(equity)
        features[0] = equity  # Raw equity (0-1)
        features[1] = equity ** 2  # Non-linearity bonus for very strong hands

        # ========== STACK (continuous) ==========
        my_position = player_data['position']
        my_stack = player_data['stack'][my_position]
        features[2] = my_stack  # Normalized stack (chips / (BB * 100))
        features[3] = np.log1p(my_stack)  # Log scale to handle wide range

        # ========== STAGE (one-hot) ==========
        stage_one_hot = community_data['stage']
        stage_idx = int(np.argmax(stage_one_hot))
        features[4] = 1.0 if stage_idx == 0 else 0.0  # preflop
        features[5] = 1.0 if stage_idx == 1 else 0.0  # flop
        features[6] = 1.0 if stage_idx == 2 else 0.0  # turn
        features[7] = 1.0 if stage_idx == 3 else 0.0  # river

        # ========== COMMUNITY POT (continuous) ==========
        c_pot = float(community_data['community_pot'])
        features[8] = c_pot  # Raw pot
        features[9] = np.log1p(c_pot)  # Log scale

        # ========== CURRENT ROUND POT (continuous) ==========
        r_pot = float(community_data['current_round_pot'])
        features[10] = r_pot  # Raw round pot
        features[11] = np.log1p(r_pot)  # Log scale
        features[12] = np.sqrt(r_pot + 1)  # Sqrt scale

        # ========== POT ODDS ==========
        # Pot odds: how much of the total pot am I risking?
        total_pot = c_pot + r_pot
        total_exposure = my_stack + r_pot
        if total_exposure > 0:
            features[13] = total_pot / (total_exposure + 1e-6)
        else:
            features[13] = 0.0

        # ========== BIAS TERM ==========
        features[15] = 1.0  # Always 1 for bias

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
        
        if len(self.hand_rewards) == 0:
            print("No action rewards data to plot yet.")
            return
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.hand_rewards, label='Action Rewards', color='blue')
        plt.xlabel('Hand Number')
        plt.ylabel('Reward')
        plt.title('Training Progress: Hand Rewards Over Time')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print(f"\n[Agent] Training plot saved as 'training_progress.png'")
        plt.show()

    def __str__(self):
        return self.name