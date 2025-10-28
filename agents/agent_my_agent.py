"""
Complete Poker Agent combining Player interface with Q-learning Agent
This file should replace your agent_my_agent.py
"""

from gym_env.env import Action
import numpy as np
import os

# Configuration
USE_TRAINED_WEIGHTS = True
FEATURE_SIZE = 30  # Increased from 21 for poker features
NUM_ACTIONS = 7    # FOLD, CHECK, CALL, RAISE_3BB, RAISE_HALF_POT, RAISE_POT, ALL_IN


class Player:
    """
    Player class that implements both the neuron_poker Player interface
    AND the Q-learning agent functionality from your blackjack code.
    """
    
    def __init__(self, epsilon, alpha, gamma, name="MyAgent", stack_size=None, weights_file: str = None):
        # ===== Player Interface (required by neuron_poker) =====
        self.name = name
        self.stack = stack_size
        self.autoplay = True  # CRITICAL: Environment will call action() automatically
        
        # ===== Agent Learning Parameters (from your blackjack agent) =====
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training_error = []
        
        # ===== Weight Initialization =====
        # 2D matrix with FEATURE_SIZE rows and NUM_ACTIONS columns [30x7]
        if weights_file and USE_TRAINED_WEIGHTS and os.path.exists(weights_file):
            self.load_weights(weights_file)
        else:
            self.weights = np.random.uniform(-0.1, 0.1, (FEATURE_SIZE, NUM_ACTIONS))
        
        # ===== Action Mapping =====
        # Consistent ordering for action-to-index conversion
        self.action_order = [
            Action.FOLD,
            Action.CHECK, 
            Action.CALL,
            Action.RAISE_3BB,
            Action.RAISE_HALF_POT,
            Action.RAISE_POT,
            Action.ALL_IN
        ]

    def action(self, action_space, observation, info):
        """
        Required method by neuron_poker environment.
        This is called when it's this player's turn.
        
        Args:
            action_space: List of valid Action enums for current state
            observation: Raw observation dict (not used directly)
            info: Dict containing processed observation under 'observation' key
            
        Returns:
            Action enum to take
        """
        # Extract the processed observation
        obs = info.get('observation', {})
        
        # Get list of valid actions
        valid = [a for a in action_space if a.value != 0]
        
        if not valid:
            return Action.FOLD
        
        # Choose action using epsilon-greedy Q-learning
        return self.choose_action_q_learning(obs, valid)

    def choose_action_q_learning(self, obs_dict, valid_actions):
        """
        Choose action using epsilon-greedy policy with Q-values.
        
        Args:
            obs_dict: Processed observation dictionary
            valid_actions: List of valid Action enums
            
        Returns:
            Action enum to take
        """
        # Epsilon-greedy: explore with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploit: choose action with highest Q-value
        q_values = self.get_q_values(obs_dict)
        
        # Filter Q-values to only valid actions
        valid_q_values = []
        valid_action_list = []
        
        for action in valid_actions:
            if action in self.action_order:
                idx = self.action_order.index(action)
                valid_q_values.append(q_values[idx])
                valid_action_list.append(action)
        
        if not valid_action_list:
            return valid_actions[0]
        
        # Choose action with highest Q-value
        best_idx = np.argmax(valid_q_values)
        return valid_action_list[best_idx]

    def extract_features(self, obs_dict):
        """
        Convert poker observation dictionary to binary/normalized feature vector.
        
        This is a SIMPLE starting implementation using equity + basic features.
        You can expand this later with more sophisticated features.
        
        Args:
            obs_dict: Processed observation dictionary from info['observation']
            
        Returns:
            Feature vector of size FEATURE_SIZE
        """
        features = np.zeros(FEATURE_SIZE)
        
        # Safety check: return zeros if not a proper dict
        if not isinstance(obs_dict, dict):
            return features
        
        # ===== Feature 0: Equity (most important feature) =====
        equity = 0.0
        eq = obs_dict.get('equity')
        if eq and isinstance(eq, dict):
            equity = eq.get('me', 0.0)
        features[0] = equity
        
        # ===== Features 1-4: Stage (one-hot encoded) =====
        stage = obs_dict.get('stage', '')
        stage_map = {'PREFLOP': 1, 'FLOP': 2, 'TURN': 3, 'RIVER': 4}
        stage_idx = stage_map.get(stage, 1)
        features[stage_idx] = 1
        
        # ===== Features 5-7: Position (one-hot encoded) =====
        position = obs_dict.get('position', 0)
        num_players = obs_dict.get('active_players', 6)
        if num_players > 0:
            # Early position
            if position < num_players / 3:
                features[5] = 1
            # Middle position
            elif position < 2 * num_players / 3:
                features[6] = 1
            # Late position
            else:
                features[7] = 1
        
        # ===== Feature 8: Pot Odds =====
        call_cost = obs_dict.get('current_bet', 0) - obs_dict.get('my_contribution', 0)
        pot_size = obs_dict.get('pot', 1)
        features[8] = call_cost / max(pot_size, 1)
        
        # ===== Feature 9: Stack Ratio (normalized by initial stack) =====
        my_stack = obs_dict.get('my_stack', 0)
        initial_stack = self.stack if self.stack else 500
        features[9] = my_stack / max(initial_stack, 1)
        
        # ===== Feature 10: Pot Size (normalized) =====
        features[10] = pot_size / max(initial_stack * 6, 1)  # 6 players max
        
        # ===== Feature 11: Investment Ratio =====
        my_contribution = obs_dict.get('my_contribution', 0)
        features[11] = my_contribution / max(pot_size, 1)
        
        # ===== Feature 12: Number of Active Players (normalized) =====
        features[12] = num_players / 6.0
        
        # ===== Features 13-19: Reserved for future expansion =====
        # You can add more features here like:
        # - Betting patterns
        # - Number of raises this round
        # - Opponent aggression indicators
        # - etc.
        
        # Remaining features (20-29) are zeros for now
        
        return features

    def get_q_values(self, obs_dict):
        """
        Get Q-values for all actions in a given state.
        
        Args:
            obs_dict: Poker observation dictionary
            
        Returns:
            Array of Q-values for each action [Q(s,0), Q(s,1), ..., Q(s,6)]
        """
        features = self.extract_features(obs_dict)
        # Resulting q_values is of shape (7,) for 7 possible actions
        q_values = np.dot(features, self.weights)
        return q_values

    def update(self, obs, action, reward, next_obs, terminated):
        """
        Update weights using Q-learning (off-policy TD control).
        
        Args:
            obs: Current observation dictionary
            action: Action enum that was taken
            reward: Reward received
            next_obs: Next observation dictionary
            terminated: Whether the episode has ended
        """
        # Extract features from current observation
        features = self.extract_features(obs)
        
        # Get current Q-values
        q_values = self.get_q_values(obs)
        
        # Map action to index
        if action not in self.action_order:
            # If action not in our ordering, skip update
            return
        
        action_idx = self.action_order.index(action)
        
        # Current Q-value prediction for the action taken
        q_prediction = q_values[action_idx]
        
        # Calculate TD target
        if terminated:
            td_target = reward
        else:
            max_next_q = np.max(self.get_q_values(next_obs))
            td_target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = td_target - q_prediction
        
        # Update weights for the action taken
        # Only features that are non-zero will contribute to the update
        self.weights[:, action_idx] += self.alpha * td_error * features
        
        # Track training error for monitoring
        self.training_error.append(td_error)

    def save_weights(self, filename: str):
        """
        Save the trained weights to a file.
        
        Args:
            filename: Path to save the weights file
        """
        np.save(filename, self.weights)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename: str):
        """
        Load pre-trained weights from a file.
        
        Args:
            filename: Path to the weights file
        """
        if os.path.exists(filename):
            self.weights = np.load(filename)
            print(f"Weights loaded from {filename}")
        else:
            print(f"Warning: Weights file {filename} not found. Using random initialization.")
            self.weights = np.random.uniform(-0.1, 0.1, (FEATURE_SIZE, NUM_ACTIONS))

    def __str__(self):
        """String representation for the environment."""
        return self.name
