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
        self.episode_history = []

        # Track if we've updated for the current hand
        self.last_update_hand = -1
        self.current_hand = 0

        # Training metrics
        self.hand_mean_squared_errors = []
        self.game_wins = 0


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

        self.episode_history.append((saved_info, action))

        self.num_actions += 1
        return action
    

    def add_history_marker(self):
        """Add a marker to indicate the end of a hand in history."""
        self.num_markers += 1
        self.episode_history.append((None, None))

    
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
        if len(self.episode_history) == 0:
            print("History is empty")
            return

        hand_idx = 0
        hand_td_errors = []  # Local accumulator for THIS hand

        for i, (info, action) in enumerate(self.episode_history):

            if info is None and action is None:
                if len(hand_td_errors) > 0:
                    mean_mse = np.mean(np.square(hand_td_errors))
                    self.hand_mean_squared_errors.append(mean_mse)
                    hand_td_errors = []
                hand_idx += 1
                continue  # skip hand start markers
            
            if i + 1 == len(self.episode_history):
                print("End of history reached")
                break

            next_info = self.episode_history[i + 1][0]
            next_action = self.episode_history[i + 1][1]

            # FIRST: Determine if this is last action of hand because a None placeholder is added at the start of each hand
            is_last_action_in_hand = (next_info is None and next_action is None)

            # Calculate reward
            if is_last_action_in_hand:
                if hand_idx >= len(funds_history):
                    print(f"[Agent] WARNING: hand_idx {hand_idx} >= len(funds_history) {len(funds_history)}")
                    break

                start_of_hand_stack = funds_history.iloc[hand_idx].iloc[my_position]
                if hand_idx + 1 < len(funds_history):
                    end_of_hand_stack = funds_history.iloc[hand_idx + 1].iloc[my_position]
                else:
                    end_of_hand_stack = self.initial_stack
                    print("[Agent] WARNING: Using initial stack for end_of_hand_stack")
                    
                terminal_reward = (end_of_hand_stack - start_of_hand_stack) / self.big_blind
                reward = terminal_reward * 10  # Scale reward

                print(f"[Agent] HAND END: hand_idx={hand_idx}, start={start_of_hand_stack}, end={end_of_hand_stack}, reward={reward}")
            else:
                equity_before = info['player_data']['equity_to_river_alive']
                equity_after = next_info['player_data']['equity_to_river_alive']
                equity_change = equity_after - equity_before
                reward = equity_change * 5  # Scale intermediate reward

                # Penalize negative equity changes more heavily to discourage poor actions
                if equity_change < 0:
                    reward *= 1.5

            # SECOND: Perform Q-learning update (computes TD error)
            self.num_updates += 1
            td_error = self._q_learning_update(info, action, reward, next_info)

            # THIRD: Collect the TD error that was just computed
            hand_td_errors.append(td_error)

            print(f"[Agent] Update {i}: Action={action}, Reward={reward}, TD Error={td_error:.4f}")

        # if agents final stack is not zero, consider it a win
        self.game_wins += 1 if funds_history.iloc[-1].iloc[my_position] > 0 else 0


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

    def _extract_features(self, info: Dict) -> np.ndarray:
        """
        Extract features optimized for Q-learning in poker.

        Design principles:
        1. Each feature should represent ONE meaningful concept
        2. Features should be normalized to similar scales (mainly [-1, 1] or [0, 1])
        3. Non-linear relationships encoded via explicit feature expansion
        4. No feature should be a linear combination of others (avoid multicollinearity)
        5. All features should have non-zero variance during training

        Total: 16 features
        """
        features = np.zeros(16, dtype=np.float64)
        player_data = info['player_data']
        community_data = info['community_data']

        # =========================================================================
        # SECTION 1: HAND STRENGTH (Features 0-2) - 3 features
        # =========================================================================
        # These capture the immediate quality of the decision point

        equity = player_data['equity_to_river_alive']
        equity = 0.0 if np.isnan(equity) else float(equity)
        equity = np.clip(equity, 0.0, 1.0)

        # Feature 0: Raw equity [0, 1]
        # Why: Direct measure of win probability. Linear relationship important.
        features[0] = equity

        # Feature 1: Equity squared [0, 1]
        # Why: Creates non-linear separation. Distinguishes weak (0.3) vs strong (0.7) hands
        #      Helps model learn that marginal improvements matter more at extremes
        features[1] = equity ** 2

        # Feature 2: Distance from 50-50 (centered) [-1, 1]
        # Why: Different learning for equity above vs below break-even
        #      Q-learner needs to differentiate when you're ahead (>0.5) vs behind (<0.5)
        features[2] = 2 * equity - 1

        # =========================================================================
        # SECTION 2: POSITION (Features 3-5) - 3 features
        # =========================================================================
        # Position fundamentally changes strategy in poker

        position = player_data['position']
        num_players = player_data.get('num_players', 6)

        # Feature 3: Button indicator [0, 1]
        # Why: Button is strongest position (last to act post-flop)
        #      Binary indicator so Q-learner clearly sees this context
        features[3] = 1.0 if position == (num_players - 1) else 0.0

        # Feature 4: Blind indicator [0, 1]
        # Why: Blinds are weakest position (act first post-flop)
        #      Requires tighter play, different Q-values
        features[4] = 1.0 if position in [0, 1] else 0.0

        # Feature 5: Position continuous [-1, 1]
        # Why: Captures gradual position strength even for middle positions
        #      Maps button=1, small blind=-1, linearly interpolates middle
        #      Helps model generalize to different player counts
        position_ratio = (position - (num_players - 1)) / (num_players - 1)  # [-1, 0]
        features[5] = position_ratio

        # =========================================================================
        # SECTION 3: STACK DEPTH (Feature 6) - 1 feature
        # =========================================================================
        # Stack depth (in big blinds) determines bet sizing and fold equity

        my_stack = player_data['stack'][position]
        stack_bb = my_stack / self.big_blind

        # Feature 6: Effective stack depth [-1, 1]
        # Why: Log scale captures both shallow (1-10 BB) and deep (100+ BB) stacks
        #      Tanh keeps it bounded, prevents extreme values from dominating updates
        #      Log1p handles edge case of 0 stack
        # 
        # Reasoning:
        #   - 1 BB: log(2) / 5 ≈ 0.14 (very short, all-in likely)
        #   - 10 BB: log(11) / 5 ≈ 0.50 (medium, limited maneuverability)
        #   - 50 BB: log(51) / 5 ≈ 0.82 (deep, full strategy)
        #   - 200+ BB: tanh caps it near 1.0
        features[6] = np.tanh(np.log1p(stack_bb) / 5.0)

        # =========================================================================
        # SECTION 4: GAME STAGE (Features 7-10) - 4 features (one-hot)
        # =========================================================================
        # Strategy fundamentally different at each stage

        stage_one_hot = community_data['stage']  # [pre-flop, flop, turn, river]
        stage_idx = int(np.argmax(stage_one_hot))

        # Features 7-10: One-hot encoded stage
        # Why: One-hot prevents artificial ordering (pre-flop < flop < turn < river)
        #      Allows Q-learner to assign completely independent strategies per stage
        for i in range(4):
            features[7 + i] = 1.0 if stage_idx == i else 0.0

        # =========================================================================
        # SECTION 5: POT & ODDS (Features 11-13) - 3 features
        # =========================================================================
        # Decision value comes from pot odds vs equity

        c_pot = float(community_data['community_pot'])
        r_pot = float(community_data['current_round_pot'])
        total_pot = c_pot + r_pot

        # Feature 11: Pot odds (total pot relative to big blind) [0, 1]
        # Why: Captures how much you could win. Normalized to big blind for scale-invariance.
        #      Cap at 1.0 because anything > 100 BB in pot is "mega-pot" (same strategy applies)
        if total_pot > 0:
            features[11] = min(total_pot / (self.big_blind * 100), 1.0)
        else:
            features[11] = 0.0

        # Feature 12: Round pot ratio [0, 1]
        # Why: Tells you if money recently went in this round (high r_pot) vs accumulated (high c_pot)
        #      Recent aggression changes equity interpretation
        if total_pot > 0:
            features[12] = r_pot / (total_pot + 1e-6)
        else:
            features[12] = 0.5  # If no pot yet, neutral value

        # Feature 13: Pot growth indicator [-1, 1]
        # Why: Compares community pot to round pot. High ratio means pots building gradually.
        #      Low ratio means current round is aggressive. Signals opponent behavior.
        if total_pot > 0:
            ratio = c_pot / (r_pot + 1e-6)
            # log transform: log(2) ≈ 0.7, log(0.5) ≈ -0.7
            # Centers around 0, bounds with tanh
            features[13] = np.tanh(np.log1p(ratio) / 3.0)
        else:
            features[13] = 0.0

        # =========================================================================
        # SECTION 6: EQUITY vs POT INTERACTION (Feature 14) - 1 feature
        # =========================================================================
        # Combine equity and pot odds into single "value signal"

        # Feature 14: Equity vs pot odds [-1, 1]
        # Why: Q-learner needs to know if equity justifies the pot odds
        #      This is the core EV calculation: compare hand strength to risk
        # 
        # Calculation: 
        #   - High equity (0.8) + Low pot odds (0.1 pot) = Positive (good call)
        #   - Low equity (0.2) + High pot odds (0.9 pot) = Negative (bad call)
        #   - Centered around 0 via (2*equity - 1) for symmetry
        equity_signal = 2 * equity - 1  # [-1, 1]
        pot_odds_signal = min(total_pot / (self.big_blind * 50), 1.0)  # [0, 1]
        features[14] = equity_signal - pot_odds_signal  # [-2, 1], then bounded
        features[14] = np.tanh(features[14])  # Bound to [-1, 1]

        # =========================================================================
        # SECTION 7: BIAS (Feature 15) - 1 feature
        # =========================================================================
        # Allows model to learn baseline action preferences
        features[15] = 1.0

        return features

        # =========================================================================
        # SUMMARY
        # =========================================================================
        # Total: 16 features
        # 
        # Feature Scale Distribution:
        #   - Most features in [-1, 1]: promotes stable gradient updates
        #   - No feature correlates strongly with others (except equity/equity^2)
        #   - Each feature has meaningful variance during gameplay
        #
        # Learning Properties:
        #   - Clear EV signals (Feature 14 combines equity and pot odds)
        #   - Stage-based strategy separation (Features 7-10)
        #   - Position context (Features 3-5)
        #   - Stack constraints (Feature 6)
        #   - Non-linear hand strength separation (Features 1, 2)
        #   - Opponent aggression signals (Features 12-13)


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