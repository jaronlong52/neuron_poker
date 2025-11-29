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
        self.hand_mean_squared_errors = []
        self.wins = 0


        # debugging
        self.num_actions = 0
        self.num_updates = 0
        self.num_markers = 0

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

        self.num_actions += 1
        return action
    

    def add_history_marker(self):
        """Add a marker to indicate the end of a hand in history."""
        self.num_markers += 1
        self.history.append((None, None))

    
    def decay_epsilon(self):
        """Decay epsilon after each hand."""
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)


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
        hand_td_errors = []  # Local accumulator for THIS hand

        for i, (info, action) in enumerate(self.history):

            if info is None and action is None:
                continue  # skip hand start markers
            
            if i + 1 == len(self.history):
                print("End of history reached")
                break

            next_info = None 

            # FIRST: Determine if this is last action of hand
            is_last_action_in_hand = self.history[i + 1][0] is None

            # Calculate reward
            if is_last_action_in_hand:
                start_of_hand_stack = funds_history.iloc[hand_count].iloc[my_position]
                if hand_count + 1 >= len(funds_history):
                    print("No more funds history")
                    break
                else:
                    end_of_hand_stack = funds_history.iloc[hand_count + 1].iloc[my_position]
                    
                reward = 1 if end_of_hand_stack > start_of_hand_stack else 0
                self.wins += reward
            else:
                player_win_prob_before_action = info['player_data']['equity_to_river_alive']
                next_info = self.history[i + 1][0]
                player_win_prob_after_action = next_info['player_data']['equity_to_river_alive']
                reward = (player_win_prob_after_action - player_win_prob_before_action) * self.big_blind

            # SECOND: Perform Q-learning update (computes TD error)
            self.num_updates += 1
            td_error = self._q_learning_update(info, action, reward, next_info)

            # THIRD: Collect the TD error that was just computed
            hand_td_errors.append(td_error)

            print(f"[Agent] Update {i}: Action={action}, Reward={reward}, TD Error={td_error:.4f}")

            # FOURTH: If hand ended, finalize the metrics
            if is_last_action_in_hand:
                # Calculate mean TD error for this completed hand
                mean_mse = np.mean(np.square(hand_td_errors))
                self.hand_mean_squared_errors.append(mean_mse)

                # Reset for next hand
                hand_td_errors = []
                hand_count += 1


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
        return td_error


    # =====================================================================
    # FEATURES
    # =====================================================================


    GOOD IMPOVEMENTS IN CLUADE CHAT



    def _extract_features(self, info: Dict) -> np.ndarray:
        features = np.zeros(FEATURE_SIZE, dtype=np.float64)
        player_data = info['player_data']
        community_data = info['community_data']

        # ========== EQUITY ==========
        equity = player_data['equity_to_river_alive']
        equity = 0.0 if np.isnan(equity) else float(equity)
        features[0] = equity  # [0, 1]
        features[1] = equity ** 2

        # ========== STACK - NORMALIZE ==========
        my_position = player_data['position']
        my_stack = player_data['stack'][my_position]
        # Instead of raw stack, normalize by big blind
        normalized_stack = min(my_stack / 100, 1.0)  # Cap at 1.0
        features[2] = normalized_stack
        features[3] = np.log1p(my_stack) / 10  # Divide by 10 to reduce scale

        # ========== STAGE ==========
        stage_one_hot = community_data['stage']
        stage_idx = int(np.argmax(stage_one_hot))
        features[4] = 1.0 if stage_idx == 0 else 0.0
        features[5] = 1.0 if stage_idx == 1 else 0.0
        features[6] = 1.0 if stage_idx == 2 else 0.0
        features[7] = 1.0 if stage_idx == 3 else 0.0

        # ========== POTS - NORMALIZE ==========
        c_pot = float(community_data['community_pot'])
        r_pot = float(community_data['current_round_pot'])

        # Normalize pots relative to big blind
        normalized_c_pot = min(c_pot / 100, 1.0)  # Cap at 1.0
        normalized_r_pot = min(r_pot / 100, 1.0)  # Cap at 1.0

        features[8] = normalized_c_pot
        features[9] = normalized_r_pot
        features[10] = np.log1p(c_pot) / 10
        features[11] = np.log1p(r_pot) / 10
        features[12] = np.sqrt(min(r_pot, 100)) / 10  # Sqrt and normalize

        # ========== POT ODDS ==========
        total_pot = c_pot + r_pot
        total_exposure = my_stack + r_pot
        if total_exposure > 0:
            features[13] = min(total_pot / (total_exposure + 1e-6), 1.0)  # Cap at 1.0
        else:
            features[13] = 0.0

        features[15] = 1.0  # Bias

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
    def plot_td_error(self):
        import matplotlib.pyplot as plt

        if len(self.hand_mean_squared_errors) == 0:
            print("No TD error data recorded.")
            return

        data = np.array(self.hand_mean_squared_errors)

        # Dynamic window: ~5-10% of data, minimum 5
        window = max(5, len(data) // 15)
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(12, 6))
        plt.plot(data, linewidth=1, alpha=0.3, label='Raw TD Error')
        plt.plot(smoothed, linewidth=2, label=f'Smoothed (window={window})')
        plt.xlabel("Hand Number")
        plt.ylabel("Mean Squared TD Error")
        plt.title("TD Error Over Training")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("td_error_progress.png", dpi=150)
        print(f"Data points: {len(self.hand_mean_squared_errors)}, Window size: {window}")
        plt.show()


    def __str__(self):
        return self.name