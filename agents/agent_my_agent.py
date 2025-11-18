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
        self.hand_rewards = []  # Track reward per hand
        self.avg_td_errors = []  # Track average TD error per hand
        self.cumulative_stack = []  # Track stack size over time

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

    def action(self, legal_actions: List[Action], observation: np.ndarray, info: Dict, funds_history: pd.DataFrame) -> Action:

        _ = observation  # not used

        stage = info["community_data"]["stage"]
        my_position = info["player_data"]["position"]
        stage_data = info["stage_data"]

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
            print(f"\n!! Call Update (Hand {self.current_hand} -> {len(funds_history)}) !!\n")
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
        print(f"\nHand: {self.current_hand}, Stage: {np.argmax(stage)}, Action: {action}\n")
        
        # Get current stack
        if len(funds_history) > 0:
            current_stack = funds_history.iloc[-1, my_position]
        else:
            current_stack = self.initial_stack

        # to avoid reference bugs
        import copy
        saved_info = copy.deepcopy(info)

        self.history.append((saved_info, action, legal_actions, current_stack))
        return action


    def _choose_greedy(self, legal_actions: List[Action], info: Dict) -> Action:
        features = self._extract_features(info)
        q_values = np.dot(features, self.weights)
        legal_q = [(a, q_values[self.action_order.index(a)]) for a in legal_actions]
        return max(legal_q, key=lambda x: x[1])[0]
    

    def _update(self, funds_history: pd.DataFrame, my_position: int):
        if len(self.history) == 0:
            return

        hand_start_stack = self.history[0][3]
        hand_end_stack = funds_history.iloc[-1, my_position]
        final_hand_reward = hand_end_stack - hand_start_stack

        self.hand_rewards.append(final_hand_reward)
        self.cumulative_stack.append(hand_end_stack)
        td_errors_this_hand = []

        print(f"Final hand result: {hand_start_stack:.1f} → {hand_end_stack:.1f} (Δ {final_hand_reward:+.2f})")

        # Loop through each action in the hand
        for i, (info, action, legal_actions, stack_before) in enumerate(self.history):
            # Immediate reward = how much we put in the pot (always ≤ 0)
            if i + 1 < len(self.history):
                stack_after = self.history[i + 1][3]
            else:
                stack_after = funds_history.iloc[-1, my_position]  # final stack
            immediate_reward = stack_after - stack_before  # e.g. -2, -6, -10

            # Future reward: only the final hand outcome, discounted backward
            steps_to_end = len(self.history) - 1 - i
            future_reward = final_hand_reward * (self.gamma ** steps_to_end)

            total_reward = immediate_reward + future_reward

            # Next state (for Q(s,a) → s')
            next_info = self.history[i + 1][0] if i + 1 < len(self.history) else None

            print(f"Update [{i}]: {action:<15} | Bet: {immediate_reward:>6.2f} | Future: {future_reward:>6.2f} | Total R: {total_reward:>6.2f}")

            td_error = self._q_learning_update(info, action, total_reward, legal_actions, next_info)
            td_errors_this_hand.append(abs(td_error))


    def _q_learning_update(self,
                           info: Dict,
                           action: Action,
                           reward: float,
                           legal_actions: List[Action],
                           next_info: Dict):
        
        # may want to do something to incorporate the initial stack size into the reward calculation
        
        features = self._extract_features(info) 
        q_values = np.dot(features, self.weights)
        a_idx = self.action_order.index(action)
        q_sa = q_values[a_idx]

        if next_info is not None:
            next_features = self._extract_features(next_info)
            next_q = np.dot(next_features, self.target_weights)
            target = reward + self.gamma * np.max(next_q)
        else:
            target = reward
        
        td_error = target - q_sa
        self.weights[:, a_idx] += self.alpha * td_error * features

        print(f"   → TD error: {td_error:6.3f} | Q({action}) was {q_sa:6.3f} → {q_sa + self.alpha * td_error * np.linalg.norm(features):6.3f}")
        
        return td_error

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
            self.target_weights = self.weights.copy()
            print(f"[Agent] Loaded weights: {filename}")
        else:
            raise FileNotFoundError(filename)

    def plot_training_progress(self):
        """Plot training metrics to visualize learning"""
        import matplotlib.pyplot as plt
        
        if len(self.hand_rewards) == 0:
            print("No training data to plot yet.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Hand rewards over time
        axes[0, 0].plot(self.hand_rewards, alpha=0.3, label='Per-hand reward')
        if len(self.hand_rewards) >= 10:
            window = min(10, len(self.hand_rewards) // 5)
            rolling_avg = pd.Series(self.hand_rewards).rolling(window=window).mean()
            axes[0, 0].plot(rolling_avg, label=f'{window}-hand moving average', linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Hand Number')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Rewards per Hand')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative stack size
        axes[0, 1].plot(self.cumulative_stack, linewidth=2)
        axes[0, 1].axhline(y=self.initial_stack, color='r', linestyle='--', alpha=0.5, label='Starting stack')
        axes[0, 1].set_xlabel('Hand Number')
        axes[0, 1].set_ylabel('Stack Size')
        axes[0, 1].set_title('Stack Size Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Average TD errors
        axes[1, 0].plot(self.avg_td_errors, alpha=0.6, label='Avg TD Error per hand')
        if len(self.avg_td_errors) >= 10:
            window = min(10, len(self.avg_td_errors) // 5)
            rolling_avg_td = pd.Series(self.avg_td_errors).rolling(window=window).mean()
            axes[1, 0].plot(rolling_avg_td, label=f'{window}-hand moving average', linewidth=2)
        axes[1, 0].set_xlabel('Hand Number')
        axes[1, 0].set_ylabel('Avg |TD Error|')
        axes[1, 0].set_title('TD Error Over Time (Lower = More Stable)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative reward
        cumulative_rewards = np.cumsum(self.hand_rewards)
        axes[1, 1].plot(cumulative_rewards, linewidth=2)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Hand Number')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Cumulative Rewards Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print(f"\n[Agent] Training plot saved as 'training_progress.png'")
        plt.show()

    def print_training_summary(self):
        """Print a summary of training statistics"""
        if len(self.hand_rewards) == 0:
            print("No training data yet.")
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total hands played: {len(self.hand_rewards)}")
        print(f"Final stack size: {self.cumulative_stack[-1]:.2f}")
        print(f"Total profit/loss: {self.cumulative_stack[-1] - self.initial_stack:+.2f}")
        print(f"\nAverage reward per hand: {np.mean(self.hand_rewards):.3f}")
        print(f"Std dev of rewards: {np.std(self.hand_rewards):.3f}")
        print(f"Win rate: {100 * np.sum(np.array(self.hand_rewards) > 0) / len(self.hand_rewards):.1f}%")
        if len(self.avg_td_errors) > 0:
            print(f"\nAverage TD error: {np.mean(self.avg_td_errors):.3f}")
        else:
            print(f"\nAverage TD error: Not enough data yet")
        
        # Recent performance (last 20%)
        recent_count = max(1, len(self.hand_rewards) // 5)
        recent_rewards = self.hand_rewards[-recent_count:]
        print(f"\nRecent performance (last {recent_count} hands):")
        print(f"  Average reward: {np.mean(recent_rewards):.3f}")
        print(f"  Win rate: {100 * np.sum(np.array(recent_rewards) > 0) / len(recent_rewards):.1f}%")
        print("="*60 + "\n")

    def __str__(self):
        return self.name