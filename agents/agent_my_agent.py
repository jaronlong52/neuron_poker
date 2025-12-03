from gym_env.env import Action, Stage
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Tuple


# Configuration
FEATURE_SIZE = 24
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
        weights_file: str = None,
        num_update_passes: int = 1
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
        self.num_update_passes = num_update_passes

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

        exploited = None
        action = None
        if not legal_actions:
            action = Action.FOLD
        elif np.random.random() < self.epsilon:
            action = np.random.choice(legal_actions)
            exploited = False
        else:
            action = self._choose_greedy(legal_actions, info)
            exploited = True
        
        source_of_decision = "greedily" if exploited else "at random"
        # print action chosen out of legal actions
        print(f"[Agent]  Chosen: {action.name} ({source_of_decision}),                   Legal: {[a.name for a in legal_actions]}, Epsilon: {self.epsilon:.4f}")

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


    def _compute_ev_reward(self, info: Dict, action: Action) -> float:
        """
        Computes an Expected Value (EV) based reward for a single action.
        This includes special EV handling for FOLD and correctly calculates the EV of a RAISE.
        """
        player = info["player_data"]
        comm = info["community_data"]
        current_stage = comm["stage"].index(True)
        stage_data = info["stage_data"][current_stage]

        position = player["position"]
        equity = float(player["equity_to_river_alive"])
        equity = max(0.0, min(1.0, equity))  # safety clamp

        # Pot sizes
        total_pot = float(comm["community_pot"]) + float(comm["current_round_pot"])

        # Contribution / call sizes
        # Safely extract current_stage data
        min_call = stage_data["min_call_at_action"][position]
        contribution = stage_data["contribution"][position]
        cost_to_call = max(0.0, min_call - contribution)
        
        # 💡 Minimal Change: Determine the true amount committed by the player for this action 💡
        money_committed = cost_to_call # Default for CALL, FOLD (since it returns 0)

        # ───────────────────────────────────────────────
        # 1. FOLD EV
        # EV(fold) = 0.0, as no further money is committed or lost/gained.
        # ───────────────────────────────────────────────
        if action == Action.FOLD:
            return 0.0

        # ───────────────────────────────────────────────
        # 2. CHECK EV
        # Check costs nothing (cost_to_call = 0). EV = equity × total_pot
        # ───────────────────────────────────────────────
        if action == Action.CHECK:
            ev = equity * total_pot
            
            # normalize + clip
            ev /= self.big_blind
            ev = np.clip(ev, -5, 5)
            return ev

        # ───────────────────────────────────────────────
        # 3. CALL / RAISE / ALL_IN EV
        # EV = equity * (pot + new_money) - (1 - equity) * new_money
        # ───────────────────────────────────────────────
        
        # We need the ACTUAL money committed beyond the current contribution.
        # This requires knowing the final bet amount the action results in.
        
        # ASSUMPTION: The 'info' dictionary contains 'action_bet_size' or 
        # a similar key that holds the total chips put in the current street 
        # by the agent *after* taking the action.
        # Since this key is not in your provided 'info', we must infer the cost 
        # based on the action name relative to the cost_to_call.
        
        if action in [Action.CALL, Action.ALL_IN]:
            # For a CALL, the money committed is simply the cost_to_call.
            # For ALL_IN, the entire stack (minus contribution) should be committed, 
            # but since we don't know the stack here, we use cost_to_call for a baseline. 
            # The terminal reward propagation is often better for ALL_IN anyway.
            money_committed = cost_to_call

        elif action.name.startswith("RAISE_"):
            # This is the crucial fix: we need to find the total chips committed 
            # for the raise. We will approximate this by the next min_raise plus 
            # the current call, or ideally, the actual amount of the raise.
            
            # Since the game engine determines the legal actions, we assume that 
            # any RAISE action is correctly costed by the environment.
            # The simplest assumption that forces differentiation between raises is:
            
            # 💡 FIX: Temporarily disable intermediate reward for large raises 
            # and rely *only* on the terminal reward, which is less risky than 
            # calculating an incorrect EV here.
            if action in [Action.RAISE_HALF_POT, Action.RAISE_POT, Action.RAISE_2POT]:
                return 0.0 # Neutral reward, letting the TD-learning propagate the true value from hand end.
            
            # Otherwise, assume RAISE_3BB is similar to a small call/bet
            if action == Action.RAISE_3BB:
                # We assume RAISE_3BB is a small increase over the cost to call.
                # To be minimal, we keep it as cost_to_call, but rely on the TD update
                # to learn the value difference based on game state changes.
                money_committed = cost_to_call 
        
        # ------------------------------------------------------------------
        # Recalculated EV Term (The formula itself is correct for a CALL, 
        # but the variable must represent the *true* additional money risked)
        # ------------------------------------------------------------------

        # If the action is a large raise and we return 0.0, we skip this section.
        if action not in [Action.RAISE_HALF_POT, Action.RAISE_POT, Action.RAISE_2POT]:
            
            # Win amount includes total pot + the money you are putting in now.
            win_amount = total_pot + money_committed 

            # The cost term is the money committed now.
            ev = (equity * win_amount) - ((1.0 - equity) * money_committed)

            # Normalize by big blind to stabilize learning
            ev /= self.big_blind

            # Cap the EV to avoid reward explosions
            ev = float(np.clip(ev, -5, 5))

            return ev
        
        return 0.0 # Return 0.0 for large raises that were filtered above


    def _choose_greedy(self, legal_actions: List[Action], info: Dict) -> Action:
        features = self._extract_features(info)
        q_values = np.dot(features, self.weights)
        legal_q = [(a, q_values[self.action_order.index(a)]) for a in legal_actions]
        return max(legal_q, key=lambda x: x[1])[0]
    

    # Called by environment at the end of each episode
    def update(self, funds_history: pd.DataFrame, my_position: int):
        if len(self.episode_history) == 0:
            print("History is empty")
            return
        
        # NEW: Dictionary to accumulate TD errors across passes for each state-action index
        cumulative_td_errors = {}  # key: state-action index, value: list of TD errors

        for pass_num in range(self.num_update_passes):
            hand_idx = 0
            hand_td_errors = []  # Local accumulator for THIS hand

            for i, (info, action) in enumerate(self.episode_history):

                if info is None and action is None:
                    if len(hand_td_errors) > 0:
                        if pass_num == self.num_update_passes - 1:  # Only record on final pass
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
                    reward = np.tanh(terminal_reward / 3)  # Scale final reward

                    if pass_num == self.num_update_passes - 1:
                        print(f"[Agent] HAND END: hand_idx={hand_idx}, start={start_of_hand_stack}, end={end_of_hand_stack}, reward={reward}")
                else:
                    reward = self._compute_ev_reward(info, action)

                # SECOND: Perform Q-learning update (computes TD error)
                self.num_updates += 1
                td_error = self._q_learning_update(info, action, reward, next_info)

                # NEW: Accumulate TD errors for this state-action pair
                if i not in cumulative_td_errors:
                    cumulative_td_errors[i] = []
                cumulative_td_errors[i].append(td_error)

                # THIRD: Collect the TD error that was just computed
                hand_td_errors.append(td_error)

                if pass_num == self.num_update_passes - 1:
                    print(f"[Agent] Update {i}: Action={action}, Reward={reward}, TD Error={td_error:.4f}")

        # NEW: After all passes, optionally log averaged TD errors
        if self.num_update_passes > 1:
            avg_td_errors = {idx: np.mean(errors) for idx, errors in cumulative_td_errors.items()}
            print(f"[Agent] Completed {self.num_update_passes} passes. Avg TD errors across passes: {np.mean(list(avg_td_errors.values())):.4f}")
            
        # if agents final stack is not zero, consider it a win
        self.game_wins += 1 if funds_history.iloc[-1].iloc[my_position] > 0 else 0
        self.num_actions = 0


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
        Extracts a feature vector from the game state info dictionary.
        This is a placeholder implementation and should be replaced with
        meaningful features based on the poker game state.
        """
        features = np.zeros(FEATURE_SIZE)

        # Player data
        player = info["player_data"]
        equity = player["equity_to_river_alive"]
        position = player["position"]
        stack = player["stack"][position]

        # Community data
        comm = info["community_data"]
        stage = comm["stage"]
        current_stage = stage.index(True)
        community_pot = comm["community_pot"]
        current_round_pot = comm["current_round_pot"]
        total_pot = community_pot + current_round_pot
        active_players = comm["active_players"]
        num_players = len(active_players)
        num_active_players = sum(active_players)

        # Stage data
        stage_data = info["stage_data"][current_stage]
        raises = stage_data["raises"]
        min_call = stage_data["min_call_at_action"][position]
        contribution = stage_data["contribution"][position]
        stack_at_action = stage_data["stack_at_action"]
        
        
        # === CORE STATE FEATURES ===
        # Probability of winning to the river.
        features[0] = equity

        # Pot Odds - Break-even threshold for calling
        cost = max(0, min_call - contribution)
        pot_odds = cost / (total_pot + cost + 1e-9)
        features[1] = pot_odds

        # Stack-to-pot ratio for deep vs short stack leverage
        spr = stack / (total_pot + 1e-9)
        features[2] = spr

        # Position normalized
        position_norm = position / max((num_players - 1), 1)
        features[3] = position_norm

        # Fraction of active players
        features[4] = num_active_players / num_players


        # === INTERACTION FEATURES ===
        # Expected value or equity - pot odds
        features[5] = equity - pot_odds

        # Equity x position - strong hands gain more value in late position
        features[6] = equity * position_norm

        # Equity x Inverse stack-to-pot ratio - strong hands dominate when SPR is small
        features[7] = equity * (1 / (spr + 1e-9))

        # Bluff catching difficulty
        features[8] = (1 - equity) * pot_odds

        # Commitment ratio
        commitment_ratio = contribution / (stack + contribution + 1e-9)
        features[9] = commitment_ratio

        # Pressure - normalized size of the bet relative to BB
        pressure = cost
        features[10] = pressure

        # Equity x pressure - strong hands respond differently to aggression than weak ones
        features[11] = equity * pressure


        # === STAGE FLAGS ===
        features[12] = 1 if current_stage == 0 else 0    # Preflop
        features[13] = 1 if current_stage == 1 else 0    # Flop
        features[14] = 1 if current_stage == 2 else 0    # Turn
        features[15] = 1 if current_stage == 3 else 0    # River


        # === HIGH VALUE STRATEGY FEATURES ===
        # Raise-to-pot ratio for how expensive a raise is relative to pot
        features[16] = cost / (total_pot + 1e-9)

        # Opponent aggression last street - whether someone else raised
        opponent_raised = any(raises[j] for j in range(len(raises)) if j != position)
        other_player_last_aggressor = 1 if opponent_raised else 0
        features[17] = other_player_last_aggressor

        # Pot growth rate - captures previous betting tempo
        features[18] = current_round_pot / max((community_pot + 1e-9), 1)

        # Stack vs average stack - pressure you can apply / pressure you are under
        active_stacks = [stack_at_action[i] for i in range(len(stack_at_action)) if active_players[i]]
        average_stack = np.mean(active_stacks) if active_stacks else 1.0
        features[19] = stack / (average_stack + 1e-9)

        # Effective stack ratio - max possible amount that can go in
        opponent_stacks = [stack_at_action[i] for i in range(len(stack_at_action)) if i != position and active_players[i]]
        if opponent_stacks:
            max_opponent_stack = max(opponent_stacks)
            effective_stack = min(stack, max_opponent_stack)
            features[20] = effective_stack / (total_pot + 1e-9)
        else:
            features[20] = 0

        # Equity x (cost / stack) - riskiness of continuing
        features[21] = equity * (cost / (stack + 1e-9))

        # SPR x position - deep stack in position = more freedom
        features[22] = spr * position_norm

        # Commitment x Aggression - detects bluff-heavy vs value-heavy pressure when still committed
        features[23] = commitment_ratio * other_player_last_aggressor

        features = np.clip(features, -10, 10) # clipping to avoid extreme values
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
    def plot_td_error(self, name: str = "td_error_progress.png"):
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
        plt.savefig(name, dpi=150)
        print(f"Data points: {len(self.hand_mean_squared_errors)}, Window size: {window}")
        plt.show()


    def __str__(self):
        return self.name