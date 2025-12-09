from gym_env.env import Action, Stage
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Tuple


# Configuration
FEATURE_SIZE = 45
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
        num_update_passes: int = 1,
        isNotLearning: bool = True,
        num_episodes: int = -1,
    ):
        # Required by environment
        self.name = name
        self.autoplay = True
        self.initial_stack = stack_size

        # Hyperparameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.alpha = alpha
        self.gamma = gamma
        self.big_blind = big_blind
        self.num_update_passes = num_update_passes
        self.isNotLearning = isNotLearning
        self.num_episodes = num_episodes

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

        if self.isNotLearning:
            print(f"[Old Model] Initialized weights: {self.weights.shape}")
        else:
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
        elif self.isNotLearning:
            action = self._choose_greedy(legal_actions, info)
            exploited = True
        elif np.random.random() < self.epsilon:
            action = np.random.choice(legal_actions)
            exploited = False
        else:
            action = self._choose_greedy(legal_actions, info)
            exploited = True
        
        source_of_decision = "greedily" if exploited else "at random"
        # print action chosen out of legal actions
        if not self.isNotLearning:
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
        self.epsilon = max(self.epsilon, self.epsilon_min)


    def _compute_ev_reward(self, info: Dict, action: Action) -> float:
        """
        Simple intermediate rewards based on equity vs pot odds.
        Kept small to let terminal rewards dominate learning.
        """
        if action == Action.FOLD:
            return 0.0

        player = info["player_data"]
        comm = info["community_data"]
        stage_data = info["stage_data"][comm["stage"].index(True)]

        equity = np.clip(float(player["equity_to_river_alive"]), 0.0, 1.0)
        position = player["position"]

        # Already normalized by (BB * 100)
        total_pot = float(comm["community_pot"]) + float(comm["current_round_pot"])
        cost = max(0.0, stage_data["min_call_at_action"][position] - stage_data["contribution"][position])

        # CHECK: Free action, small equity-based reward
        if action == Action.CHECK:
            return equity * 0.1  # Max reward: 0.1 (very small)

        # All other actions: Simple pot odds comparison
        pot_odds = cost / (total_pot + cost + 1e-9)
        ev_signal = equity - pot_odds

        # Scale to [-0.2, 0.2] range - much smaller than terminal rewards
        return np.clip(ev_signal * 0.5, -0.2, 0.2)


    def _choose_greedy(self, legal_actions: List[Action], info: Dict) -> Action:
        features = self._extract_features(info)
        q_values = np.dot(features, self.weights)
        legal_q = [(a, q_values[self.action_order.index(a)]) for a in legal_actions]
        return max(legal_q, key=lambda x: x[1])[0]
    

    # Called by environment at the end of each episode
    def update(self, funds_history: pd.DataFrame, my_position: int):
        if self.isNotLearning:
            self.num_actions = 0
            return

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

        if abs(td_error) > 10:
            print(f"   Large TD error: {td_error:.2f}")
            print(f"   Q(s,a): {q_sa:.2f}, Target: {target:.2f}")
            print(f"   Feature max: {np.max(np.abs(features)):.2f}")
            print(f"   Weight max: {np.max(np.abs(self.weights[:, a_idx])):.2f}")
        
        # clip TD error to prevent exploding updates
        td_error = np.clip(td_error, -3, 3)
        # Update with clipped gradient
        gradient = self.alpha * td_error * features
        gradient = np.clip(gradient, -0.5, 0.5)
        self.weights[:, a_idx] += gradient
        self.weights = np.clip(self.weights, -10, 10)  # prevent weight explosion
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
        BB = self.big_blind # Added for normalization consistency

        # Player data
        player = info["player_data"]
        
        # equity_to_river_alive:
        # Estimated win probability (0.0 – 1.0) of the current player's hand
        # against all remaining active opponents, assuming:
        # - the hand goes to showdown (all board cards are dealt),
        # - opponents have random hole cards from the remaining deck,
        # - folded players are excluded.
        # Computed via Monte-Carlo simulation (1000 iterations).
        # This is the agent's primary signal of current hand strength vs. the field.
        equity = player["equity_to_river_alive"]

        position = player["position"]
        stack = player["stack"][position]

        # Community data
        comm = info["community_data"]
        stage = comm["stage"]
        current_stage = stage.index(True) # 0: Preflop, 1: Flop, 2: Turn, 3: River
        community_pot = comm["community_pot"]
        current_round_pot = comm["current_round_pot"]
        total_pot = community_pot + current_round_pot
        active_players = comm["active_players"]
        num_players = len(active_players)
        num_active_players = sum(active_players)

        # Stage data
        all_stage_data = info["stage_data"]
        current_stage_data = all_stage_data[current_stage]
        raises = current_stage_data["raises"]
        min_call = current_stage_data["min_call_at_action"][position]
        contribution = current_stage_data["contribution"][position]
        stack_at_action = current_stage_data["stack_at_action"]
        
        # Cost to call
        cost = max(0, min_call - contribution)

        # Convert to BB units for human-readable calculations
        stack_bb = stack * 100  # Now in BB units (0.5 -> 50 BB)
        total_pot_bb = total_pot * 100  # Now in BB units
        cost_bb = cost * 100  # Now in BB units
        
        # === CORE STATE FEATURES ===
        # Probability of winning to the river.
        features[0] = equity

        # Pot Odds - Break-even threshold for calling
        pot_odds = cost / (total_pot + cost + 1e-9)
        features[1] = pot_odds

        # Stack-to-pot ratio for deep vs short stack leverage
        spr = stack_bb / (total_pot_bb + 1e-9)
        features[2] = np.clip(spr / 50, 0, 1)

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
        features[7] = equity * np.clip(1 / (spr + 1e-9), 0, 1)

        # Bluff catching difficulty
        features[8] = (1 - equity) * pot_odds

        # Commitment ratio
        commitment_ratio = contribution / (stack + contribution + 1e-9)
        features[9] = commitment_ratio

        # Pressure - normalized size of the bet relative to BB
        features[10] = np.clip(cost_bb / 5, 0, 1)

        # Equity x pressure - strong hands respond differently to aggression than weak ones
        features[11] = equity * features[10]


        # === STAGE FLAGS ===
        features[12] = 1 if current_stage == 0 else 0 # Preflop
        features[13] = 1 if current_stage == 1 else 0 # Flop
        features[14] = 1 if current_stage == 2 else 0 # Turn
        features[15] = 1 if current_stage == 3 else 0 # River


        # === HIGH VALUE STRATEGY FEATURES ===
        # Raise-to-pot ratio for how expensive a raise is relative to pot
        rtp_ratio = cost / (total_pot + 1e-9)
        features[16] = np.clip(rtp_ratio, 0, 1)

        # Opponent aggression last street - whether someone else raised
        opponent_raised = any(raises[j] for j in range(len(raises)) if j != position)
        features[17] = 1 if opponent_raised else 0

        # Pot growth rate - captures previous betting tempo
        features[18] = np.clip(current_round_pot / (community_pot + 1e-9), 0, 5) / 5

        # Stack vs average stack - pressure you can apply / pressure you are under
        active_stacks = [stack_at_action[i] for i in range(len(stack_at_action)) if active_players[i]]
        average_stack = np.mean(active_stacks) if active_stacks else stack
        features[19] = np.clip(stack / (average_stack + 1e-9), 0, 2) / 2

        # Effective stack ratio - max possible amount that can go in
        opponent_stacks = [stack_at_action[i] for i in range(len(stack_at_action)) if i != position and active_players[i]]
        if opponent_stacks:
            max_opponent_stack = max(opponent_stacks)
            effective_stack = min(stack, max_opponent_stack)
            effective_stack_bb = effective_stack * 100  # in BB units
            features[20] = np.clip(effective_stack_bb / (total_pot_bb + 1e-9), 0, 20) / 20
        else:
            features[20] = 0

        # Equity x (cost / stack) - riskiness of continuing
        features[21] = equity * np.clip(cost / (stack + 1e-9), 0, 1)

        # SPR x position - deep stack in position = more freedom
        features[22] = features[2] * position_norm

        # Commitment x Aggression - detects bluff-heavy vs value-heavy pressure when still committed
        features[23] = commitment_ratio * features[17]


        # === OPPONENT MODELING ===
        # Get list of opponent positions (exclude our position, only include active players)
        opponent_positions = [i for i in range(num_players) if i != position and active_players[i]]

        if opponent_positions:
            # Feature[24]: Average opponent aggression THIS STREET ONLY
            # How many opponents raised on the current street?
            opponents_who_raised_this_street = sum(1 for opp_pos in opponent_positions 
                                                   if current_stage_data["raises"][opp_pos])
            # Normalize by number of opponents
            features[24] = opponents_who_raised_this_street / len(opponent_positions)
            # Interpretation: 0 = no opponents raised, 1 = all opponents raised this street

            # Feature[25]: Total opponent aggression ACROSS ALL STREETS
            # Count how many times ANY opponent raised in the entire hand so far
            total_opponent_raises = 0
            for street_idx in range(current_stage + 1):  # All streets up to current
                for opp_pos in opponent_positions:
                    if all_stage_data[street_idx]["raises"][opp_pos]:
                        total_opponent_raises += 1

            # Normalize by (number of streets played * number of opponents)
            # This gives us "raises per opponent per street"
            features[25] = total_opponent_raises / (len(opponent_positions) * (current_stage + 1))
            # Interpretation: 0 = opponents never raised, 1 = every opponent raised every street

            # Opponent calling frequency (passive players)
            total_opponent_calls = sum(current_stage_data["calls"][i] 
                                   for i in opponent_positions if i < len(current_stage_data["calls"]))
            features[26] = total_opponent_calls / len(opponent_positions)
        else:
            features[24] = 0  # No opponents left (heads up and won, or everyone folded)
            features[25] = 0
            features[26] = 0


        # === BLUFFING INDICATORS ===
        # Bluff profitability = low equity + few opponents + already committed
        # (Bluffs work better when fewer players to fold out)
        bluff_opportunity = (1 - equity) * (1 / (num_active_players + 1e-9)) * commitment_ratio
        features[27] = np.clip(bluff_opportunity, 0, 1)

        # Fold equity estimate based on pot size relative to opponent stacks
        # Large bets relative to opponent stacks = more fold pressure
        if opponent_positions:
            avg_opponent_stack = np.mean([stack_at_action[i] for i in opponent_positions])
            fold_pressure = cost / (avg_opponent_stack + 1e-9)
            features[28] = np.clip(fold_pressure, 0, 2) / 2
        else:
            features[28] = 0

        # Semi-bluff indicator: moderate equity (draws) + aggression opportunity
        # Equity between 0.3-0.5 is often a draw
        is_draw_range = 1 if 0.25 < equity < 0.55 else 0
        # Have a draw (is_draw_range = 1), not too committed, and in good (high) position
        features[29] = is_draw_range * (1 - commitment_ratio) * position_norm


        # === ACTION HISTORY (WHAT HAVE I DONE THIS HAND?) ===
        # We need to look at ALL streets played so far (preflop through current street)
        streets_played = current_stage + 1  # +1 because stage is 0-indexed

        # Initialize counters
        agent_total_checks = 0
        agent_total_calls = 0
        agent_total_raises = 0

        # Count our actions across all streets
        for street_idx in range(streets_played):
            # Did we call on this street?
            if all_stage_data[street_idx]["calls"][position]:
                agent_total_calls += 1

            # Did we raise on this street?
            if all_stage_data[street_idx]["raises"][position]:
                agent_total_raises += 1

            # Did we check? (neither called nor raised, and we had the option to check)
            # Note: This is approximate since we don't track checks explicitly
            # We infer: if we didn't call or raise, and we are active, we must have checked
            did_not_call_or_raise = not all_stage_data[street_idx]["calls"][position] and \
                                    not all_stage_data[street_idx]["raises"][position]
            if did_not_call_or_raise:
                agent_total_checks += 1

        # Feature[30]: Checking frequency (passive play)
        # How often do we check per street?
        features[30] = agent_total_checks / streets_played
        # Interpretation: 0 = never checked, 1 = checked every street (very passive)

        # Feature[31]: Calling frequency (passive play)
        # How often do we call per street?
        features[31] = agent_total_calls / streets_played
        # Interpretation: 0 = never called, 1 = called every street (chasing)

        # Feature[32]: Raising frequency (aggressive play)
        # How often do we raise per street?
        features[32] = agent_total_raises / streets_played
        # Interpretation: 0 = never raised, 1 = raised every street (very aggressive)

        # Feature[33]: Did we show strength earlier?
        # Have we raised on ANY previous street (not including current street)?
        raised_on_previous_streets = 0
        for street_idx in range(current_stage):  # Only previous streets, not current
            if all_stage_data[street_idx]["raises"][position]:
                raised_on_previous_streets = 1
                break  # We only need to know if we did it at least once
            
        features[33] = raised_on_previous_streets
        # Interpretation: 0 = we've been passive so far, 1 = we showed strength earlier
        # Why it matters: If you raised preflop, opponents expect you to have a good hand

        # Feature[34]: Check-raise opportunity
        # Did we CHECK earlier this street, and is an OPPONENT now BETTING/RAISING?

        # Did we check this street? (we didn't call or raise yet this street)
        we_checked_this_street = (not current_stage_data["calls"][position] and 
                                  not current_stage_data["raises"][position])

        # Did any opponent raise this street?
        opponent_raised_this_street = any(current_stage_data["raises"][i] 
                                          for i in opponent_positions 
                                          if i < len(current_stage_data["raises"]))

        # Check-raise opportunity = we checked AND opponent bet
        check_raise_opportunity = we_checked_this_street and opponent_raised_this_street
        features[34] = 1 if check_raise_opportunity else 0
        # Interpretation: 1 = perfect check-raise spot (we checked, opponent bet)
        # Why it matters: Check-raising is deceptive and builds bigger pots with strong hands

        # Feature[35]: Check-raise VALUE
        # If we have a check-raise opportunity AND strong equity, this is very valuable
        features[35] = equity * features[34]
        # Interpretation: High value (>0.7) = strong hand + check-raise opportunity = RAISE NOW!
        # Why it matters: Check-raising with strong hands wins more money


        # === DECEPTION & TABLE IMAGE ===
        # Tight image: if we've folded/checked most of the time, bluffs are more credible
        total_actions = agent_total_checks + agent_total_calls + agent_total_raises + 1e-9
        features[36] = (agent_total_checks + agent_total_calls) / total_actions

        # Deception opportunity: showed strength but now facing resistance
        # (can we fold a hand we represented as strong?)
        if current_stage > 0:
            raised_last_street = all_stage_data[current_stage - 1]["raises"][position]
        else:
            raised_last_street = False
        recent_strength = 1 if raised_last_street else 0
        overall_strength = (agent_total_raises / streets_played) if streets_played > 0 else 0

        # Weight recent strength more heavily (70% recent, 30% overall)
        apparent_strength = 0.7 * recent_strength + 0.3 * overall_strength
        features[37] = apparent_strength * (1 - equity) * features[17]

        # Trap opportunity: strong hand + passive play so far
        # (slowplay strong hands to induce bluffs)
        deep_stack_flag = 1 if features[2] > 0.5 else 0  # SPR > 25
        features[38] = equity * features[36] * deep_stack_flag


        # === STAGE PROGRESSION FEATURES ===
        # Contribution increase from previous streets
        prev_contribution = sum(all_stage_data[i]["contribution"][position] for i in range(current_stage))
        current_contribution = current_stage_data["contribution"][position]
        
        # Normalize contributions by BB before calculating acceleration
        features[39] = np.clip(current_contribution / (prev_contribution + 1e-9), 0, 5) / 5

        # Pot growth across streets (betting tempo)
        prev_street_pot = sum(all_stage_data[i]["community_pot_at_action"][position] for i in range(current_stage)) if current_stage > 0 else 0
        
        # Minimal Change: Normalize pot by BB before calculating acceleration
        features[40] = np.clip(community_pot / (prev_street_pot + 1e-9), 0, 5) / 5

        # Late stage big bet indicator (turn/river aggression is more credible)
        late_stage = 1 if current_stage >= 2 else 0 # Turn or River
        features[41] = late_stage * features[16]


        # === POSITIONAL DYNAMICS ===
        # In position aggression (opponents acted before us, we close action)
        # More players acted before = more info = better position
        players_acted_before = sum(1 for i in range(position) if active_players[i])
        positional_advantage = players_acted_before / (num_active_players + 1e-9)
        features[42] = positional_advantage * position_norm

        # Feature[43]: Out of position vulnerability (0.0 – 1.0) - Added for full set
        players_to_act_after = sum(1 for i in range(position + 1, num_players) if active_players[i])
        vulnerability = players_to_act_after / (num_active_players + 1e-9)
        features[43] = vulnerability * (1 - position_norm)

        # Feature[44]: Total Pot (Normalized by BB) - Added for full set
        # 💡 Minimal Change: Adding final feature, normalized by BB
        features[44] = np.clip(total_pot_bb / 50, 0, 1)
        
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
        import pandas as pd
        import numpy as np

        if len(self.hand_mean_squared_errors) == 0:
            print("No TD error data recorded.")
            return

        # --- 1. Data Preparation ---
        data_series = pd.Series(self.hand_mean_squared_errors)

        # Calculate window size based on data length
        # Assuming self.hand_mean_squared_errors length is 9300 for the plot example
        window = max(5, len(data_series) // 15)

        # Calculate Rolling Average (Smoothed MSE)
        # The .mean() will produce NaN for the first (window - 1) points
        smoothed = data_series.rolling(window=window).mean()

        # --- 2. Plotting ---
        plt.figure(figsize=(12, 6))

        # Plot Raw MSE (Faded)
        plt.plot(data_series.index, data_series.values, linewidth=1, alpha=0.3, label='Raw Hand MSE')

        # Plot Smoothed Rolling Average
        plt.plot(smoothed.index, smoothed.values, linewidth=2, color='blue', label=f'Rolling Average MSE (Window={window})')

        # --- 3. Start/End MSE Annotations (New Feature) ---

        # Find the first non-NaN value and its index (where the rolling average starts)
        start_index = smoothed.first_valid_index()
        end_index = smoothed.index[-1]

        start_mse = smoothed.loc[start_index]
        end_mse = smoothed.loc[end_index]

        # Marker for Start MSE
        plt.plot(start_index, start_mse, marker='o', markersize=10, color='blue', linestyle='', markeredgecolor='white', markeredgewidth=1)
        
        # enforce minimum x-offset so label never appears offscreen
        safe_start_x = max(start_index, 5)

        plt.annotate(
            f"Start MSE: ~{start_mse:.2f}",
            (safe_start_x, start_mse),
            textcoords="offset points",
            xytext=(10, 20),   # positive x offset so it's inside figure
            ha="left",
            color="blue",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='blue')
        )

        # Marker for End MSE
        plt.plot(end_index, end_mse, marker='o', markersize=10, color='blue', linestyle='', markeredgecolor='white', markeredgewidth=1)
        plt.annotate(
            f'End MSE: ~{end_mse:.2f}', 
            (end_index, end_mse), 
            textcoords="offset points", 
            xytext=(-40, 20), # Offset text position
            ha='right', 
            color='blue', 
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='blue')
        )

        # --- 4. Training Session Info (New Feature) ---
        training_info = f"""
                        Episodes: {self.num_episodes}
                        Epsilon: 1.0 > {self.epsilon_min} (decay={self.epsilon_decay})
                        Alpha (LR): {self.alpha}
                        Gamma-Discount: {self.gamma}
                        Passes: {self.num_update_passes}
                        """ 
        
        # --- Title (Moved Up and Left) ---
        plt.title(
            "TD Error Over Training: Convergence Trend",
            x=0.02,      # push title left
            y=1.02,      # push title upward
            loc='left'
        )

        # Reserve upper space for title & info box
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        # --- Training Info Box (in reserved header section) ---
        plt.figtext(
            0.98,           # near right border
            0.955,          # above title, inside header area
            training_info.strip(),
            fontsize=10,
            family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.85),
            ha='right',     # align the right edge
            va='top'
        )

        # --- 5. General Plot Configuration ---
        plt.xlabel("Hand Number", y=-0.5)
        plt.ylabel("Mean Squared TD Error")
        plt.legend()
        plt.grid(True, alpha=0.3) 

        # Increase padding/spacing (New Feature)
        plt.tight_layout(pad=3.0) 

        plt.savefig(name, dpi=150)
        print(f"Data points: {len(self.hand_mean_squared_errors)}, Window size: {window}")
        plt.show()


    def __str__(self):
        return self.name