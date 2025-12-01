from gym_env.env import Action, Stage
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Tuple


# Configuration
FEATURE_SIZE = 64
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
        This includes special EV handling for FOLD and CHECK.
        """
        player = info["player_data"]
        comm = info["community_data"]
        stage_data = info["stage_data"]

        position = player["position"]
        equity = float(player["equity_to_river_alive"])
        equity = max(0.0, min(1.0, equity))  # safety clamp

        # Pot sizes
        total_pot = float(comm["community_pot"]) + float(comm["current_round_pot"])

        # Contribution / call sizes
        if isinstance(stage_data, list):
            current_stage = stage_data[-1]
        elif isinstance(stage_data, dict):
            current_stage = stage_data
        else:
            raise ValueError(f"Unexpected stage_data type: {type(stage_data)}")

        min_call = current_stage["min_call_at_action"][position]
        contribution = current_stage["contribution"][position]
        cost_to_call = max(0.0, min_call - contribution)

        # ───────────────────────────────────────────────
        # 1. FOLD EV
        # EV(fold) = 0 because folding locks in zero additional loss/gain
        # ───────────────────────────────────────────────
        if action == Action.FOLD:
            return 0.0

        # ───────────────────────────────────────────────
        # 2. CHECK EV
        # Check costs nothing; EV = equity × total_pot
        # No expected loss term because cost_to_call = 0
        # ───────────────────────────────────────────────
        if action == Action.CHECK:
            ev = equity * total_pot

            # normalize + clip
            ev /= self.big_blind
            ev = np.clip(ev, -5, 5)
            return ev

        # ───────────────────────────────────────────────
        # 3. CALL / RAISE EV
        #
        # EV = equity * win_amount − (1 − equity) * cost_to_call
        #
        # win_amount includes current pot + the additional amount you must call.
        # ───────────────────────────────────────────────
        win_amount = total_pot + cost_to_call

        ev = (equity * win_amount) - ((1.0 - equity) * cost_to_call)

        # Normalize by big blind to stabilize learning
        ev /= self.big_blind

        # Cap the EV to avoid reward explosions
        ev = float(np.clip(ev, -5, 5))

        return ev


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
        64-feature extractor using only PlayerData, StageData and CommunityData fields
        exposed by the environment (no environment changes required).
        """

        import numpy as np

        # Short names for containers (info carries dicts created in env._get_environment)
        player = info.get("player_data", {})
        comm = info.get("community_data", {})
        stage_data_list = info.get("stage_data", [])  # list of dicts (each StageData.__dict__)

        # Initialize feature vector
        features = np.zeros(64, dtype=np.float64)
        i = 0  # feature index

        # ------------------------
        # Helpers
        # ------------------------
        def safe_get(d, k, default=0.0):
            return d.get(k, default) if isinstance(d, dict) else default

        def safe_list_get(lst, idx, default=0.0):
            try:
                return lst[idx]
            except Exception:
                return default

        def clip01(x):
            return float(max(0.0, min(1.0, x)))

        def clipm1p1(x):
            return float(max(-1.0, min(1.0, x)))

        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        # ------------------------
        # Basic available values (with safe defaults)
        # ------------------------
        position = int(safe_get(player, "position", 0))
        num_players = int(safe_get(player, "num_players", len(safe_get(comm, "current_player_position", [])) or 6))

        # stacks: environment sets player_data.stack to list of stacks (normalized in env)
        stacks = safe_get(player, "stack", None)
        if stacks is None or not hasattr(stacks, "__len__"):
            # fallback: assume even stacks
            stacks = [1.0] * num_players

        # equity values (may be NaN)
        equity_alive = safe_float(safe_get(player, "equity_to_river_alive", 0.0))
        if np.isnan(equity_alive):
            equity_alive = 0.0
        equity_2 = safe_float(safe_get(player, "equity_to_river_2plr", np.nan))
        equity_3 = safe_float(safe_get(player, "equity_to_river_3plr", np.nan))

        # community pot and current round pot (env normalizes these; use as-is)
        community_pot = safe_float(safe_get(comm, "community_pot", 0.0))
        current_round_pot = safe_float(safe_get(comm, "current_round_pot", 0.0))
        total_pot = community_pot + current_round_pot + 1e-9

        big_blind = safe_float(safe_get(comm, "big_blind", getattr(self, "big_blind", 1.0)))
        small_blind = safe_float(safe_get(comm, "small_blind", getattr(self, "small_blind", 0.5)))

        # legal moves (list of booleans in comm)
        legal_moves = safe_get(comm, "legal_moves", [])

        # active players mask (list of booleans)
        active_players_mask = safe_get(comm, "active_players", [True] * num_players)
        active_count = sum(1 for x in active_players_mask if x)

        # stage one-hot vector
        stage_onehot = safe_get(comm, "stage", [0.0, 0.0, 0.0, 0.0])
        stage_idx = int(np.argmax(stage_onehot)) if any(stage_onehot) else 0

        # stage_data_list is a list of dicts from env._get_environment()
        # Each stage_data dict should contain keys: calls, raises, min_call_at_action, contribution, stack_at_action, community_pot_at_action
        # We'll aggregate across relevant entries safely.
        # ------------------------
        # A. Equity & hand-strength proxies (8)
        # ------------------------
        # 0. raw equity [0,1]
        features[i] = clip01(equity_alive); i += 1

        # 1. equity squared [0,1]
        features[i] = clip01(equity_alive * equity_alive); i += 1

        # 2. centered equity [-1,1]
        features[i] = clipm1p1(2.0 * equity_alive - 1.0); i += 1

        # 3. two-player equity if available else same as alive
        features[i] = clip01(0.0 if np.isnan(equity_2) else equity_2) ; i += 1

        # 4. three-player equity if available else same as alive
        features[i] = clip01(0.0 if np.isnan(equity_3) else equity_3); i += 1

        # 5. binary strong equity flag (above 0.66)
        features[i] = 1.0 if equity_alive > 0.66 else 0.0; i += 1

        # 6. binary weak equity flag (below 0.33)
        features[i] = 1.0 if equity_alive < 0.33 else 0.0; i += 1

        # 7. draw-proxy: if equity between 0.33 and 0.66 (possible draw)
        features[i] = 1.0 if 0.33 <= equity_alive <= 0.66 else 0.0; i += 1

        # ------------------------
        # B. Position features (6)
        # ------------------------
        # 8. is button
        features[i] = 1.0 if position == (num_players - 1) else 0.0; i += 1

        # 9. is in blinds (SB or BB)
        features[i] = 1.0 if position in (0, 1) else 0.0; i += 1

        # 10. position normalized [0,1]
        features[i] = position / max(1, (num_players - 1)); i += 1

        # 11. position centered [-1,1]
        features[i] = (position / max(1, (num_players - 1))) * 2.0 - 1.0; i += 1

        # 12. players behind normalized [0,1]
        features[i] = (num_players - 1 - position) / max(1, (num_players - 1)); i += 1

        # 13. players ahead normalized [0,1]
        features[i] = position / max(1, (num_players - 1)); i += 1

        # ------------------------
        # C. Stack & pot geometry (8)
        # ------------------------
        # stack for this agent (note: env previously normalized stacks to big_blind * 100; use as given)
        my_stack = safe_float(safe_list_get(stacks, position, 0.0))

        # 14. stack proxy (log1p then tanh)
        features[i] = clipm1p1(np.tanh(np.log1p(my_stack + 1e-9) / 5.0)); i += 1

        # 15. stack normalized by average stack
        avg_stack = sum(safe_float(s) for s in stacks) / max(1, len(stacks))
        features[i] = clip01(my_stack / (avg_stack + 1e-9)); i += 1

        # 16. stack / total_pot (shows commitment level)
        features[i] = clip01(my_stack / (total_pot + 1e-9)); i += 1

        # 17. stack-to-pot ratio (SPR) proxy (bounded)
        features[i] = clipm1p1(np.tanh(np.log1p((my_stack + 1e-9) / (total_pot + 1e-9)) / 2.0)); i += 1

        # 18. max opponent stack normalized vs my stack (threat)
        opp_max_stack = max([safe_float(s) for idx, s in enumerate(stacks) if idx != position] or [0.0])
        features[i] = clip01(opp_max_stack / (my_stack + 1e-9)); i += 1

        # 19. min opponent stack normalized vs my stack
        opp_min_stack = min([safe_float(s) for idx, s in enumerate(stacks) if idx != position] or [my_stack])
        features[i] = clip01(opp_min_stack / (my_stack + 1e-9)); i += 1

        # 20. stddev of stacks normalized
        try:
            features[i] = clip01(np.std(np.array([float(s) for s in stacks])) / (np.mean(np.array([float(s) for s in stacks])) + 1e-9))
        except Exception:
            features[i] = 0.0
        i += 1

        # ------------------------
        # D. Pot odds & call cost proxies (6)
        # ------------------------
        # Determine cost_to_call from current stage_data entry (use latest relevant entry if present)
        cost_to_call = 0.0
        my_contribution = 0.0
        # stage_data_list is a list of stage snapshots; choose the most recent (last) for best approximation
        if isinstance(stage_data_list, (list, tuple)) and len(stage_data_list) > 0:
            latest_stage = stage_data_list[-1]
            # latest_stage expected keys: min_call_at_action, contribution
            min_call_list = latest_stage.get("min_call_at_action", None)
            contrib_list = latest_stage.get("contribution", None)
            if min_call_list is not None and contrib_list is not None:
                try:
                    min_call = safe_float(min_call_list[position])
                    my_contribution = safe_float(contrib_list[position])
                    cost_to_call = max(0.0, min_call - my_contribution)
                except Exception:
                    cost_to_call = 0.0
                    my_contribution = 0.0
        # 21. cost_to_call normalized by total pot
        features[i] = clip01(cost_to_call / (total_pot + 1e-9)); i += 1

        # 22. pot odds (cost / (pot + cost))
        features[i] = clip01(cost_to_call / (total_pot + cost_to_call + 1e-9)); i += 1

        # 23. call fraction vs my stack
        features[i] = clip01(cost_to_call / (my_stack + 1e-9)); i += 1

        # 24. my contribution / total pot
        features[i] = clip01(my_contribution / (total_pot + 1e-9)); i += 1

        # 25. normalized current round pot
        features[i] = clip01(current_round_pot / (max(1.0, big_blind) * 100.0 + 1e-9)); i += 1

        # ------------------------
        # E. Stage & recent action aggregates (8)
        # ------------------------
        # 26-29 one-hot stage (preflop, flop, turn, river)
        for s in range(4):
            features[i] = 1.0 if stage_idx == s else 0.0
            i += 1

        # aggregate calls/raises across the entire stage_data_list to produce history
        total_calls = 0
        total_raises = 0
        total_contrib = 0.0
        total_mincall = 0.0
        entries_count = 0
        for sd in stage_data_list:
            try:
                calls = sd.get("calls", [])
                raises = sd.get("raises", [])
                contribs = sd.get("contribution", [])
                mincalls = sd.get("min_call_at_action", [])
                total_calls += sum(1 for x in calls if x)
                total_raises += sum(1 for x in raises if x)
                total_contrib += sum([safe_float(x) for x in contribs]) if contribs else 0.0
                total_mincall += sum([safe_float(x) for x in mincalls]) if mincalls else 0.0
                entries_count += 1
            except Exception:
                continue

        # 30. total calls normalized
        features[i] = clip01(total_calls / (num_players * max(1, entries_count))); i += 1
        # 31. total raises normalized
        features[i] = clip01(total_raises / (num_players * max(1, entries_count))); i += 1
        # 32. avg contribution per entry normalized by total_pot
        features[i] = clip01((total_contrib / max(1, entries_count)) / (total_pot + 1e-9)); i += 1
        # 33. avg min_call seen normalized
        features[i] = clip01((total_mincall / max(1, entries_count)) / (max(1.0, big_blind) * 100.0)); i += 1

        # 34. recent aggression flag (any raises anywhere)
        features[i] = 1.0 if total_raises > 0 else 0.0; i += 1

        # 35. recent calling flag
        features[i] = 1.0 if total_calls > 0 else 0.0; i += 1

        # 36. round pot ratio: current_round_pot / total_pot
        features[i] = clip01(current_round_pot / (total_pot + 1e-9)); i += 1

        # ------------------------
        # F. Active players & simple opponent proxies (8)
        # ------------------------
        # 37. active player count (normalized)
        features[i] = clip01(active_count / float(num_players)); i += 1

        # 38. folded count normalized
        features[i] = clip01((num_players - active_count) / float(num_players)); i += 1

        # 39. fraction of opponents with stack < my_stack (short stacks)
        shorter_count = sum(1 for idx, s in enumerate(stacks) if idx != position and safe_float(s) < my_stack)
        features[i] = clip01(shorter_count / max(1, (num_players - 1))); i += 1

        # 40. fraction of opponents with stack > my_stack (big stacks)
        bigger_count = sum(1 for idx, s in enumerate(stacks) if idx != position and safe_float(s) > my_stack)
        features[i] = clip01(bigger_count / max(1, (num_players - 1))); i += 1

        # 41. average opponent stack normalized by my stack
        opp_stacks = [safe_float(s) for idx, s in enumerate(stacks) if idx != position]
        avg_opp_stack = (sum(opp_stacks) / max(1, len(opp_stacks))) if opp_stacks else my_stack
        features[i] = clip01(avg_opp_stack / (my_stack + 1e-9)); i += 1

        # 42. max opponent stack normalized
        features[i] = clip01((max(opp_stacks) if opp_stacks else my_stack) / (my_stack + 1e-9)); i += 1

        # 43. opponent stack std dev normalized by mean
        try:
            features[i] = clip01(np.std(np.array(opp_stacks)) / (np.mean(np.array(opp_stacks)) + 1e-9)) if opp_stacks else 0.0
        except Exception:
            features[i] = 0.0
        i += 1

        # 44. threat indicator: max_opp_stack / total_pot (how much opponent can commit)
        features[i] = clip01((max(opp_stacks) if opp_stacks else 0.0) / (total_pot + 1e-9)); i += 1

        # ------------------------
        # G. Legal-move and action-affordance features (8)
        # ------------------------
        # Use comm['legal_moves'] (list indexed by Action enum) — environment sets it earlier
        # We'll try to be robust: check for items by name if possible otherwise fallback to common indices
        lm = legal_moves if isinstance(legal_moves, (list, tuple)) else []

        # Helper to map by presence (some envs use booleans list)
        # We'll create a few intuitive flags: can_call, can_check, can_raise_3bb, can_all_in
        can_call = False
        can_check = False
        can_raise_3bb = False
        can_all_in = False

        try:
            # If list of booleans aligned with Action enum, we can detect by index
            # Fallback: search for string names (some envs might store names)
            # As a robust approach, treat truthy elements as "some legal moves exist" and look for likely positions:
            # - If length >= 3, assume indices for CALL/CHECK/ALL_IN exist; use heuristic
            if len(lm) >= 1:
                # heuristic: CALL is often present when not max pot contributor
                can_call = bool(lm[Action.CALL.value]) if len(lm) > Action.CALL.value else any(lm)
            if len(lm) > Action.CHECK.value:
                can_check = bool(lm[Action.CHECK.value]) if len(lm) > Action.CHECK.value else False
            # RAISE_3BB and ALL_IN detection
            if len(lm) > Action.RAISE_3BB.value:
                can_raise_3bb = bool(lm[Action.RAISE_3BB.value])
            if len(lm) > Action.ALL_IN.value:
                can_all_in = bool(lm[Action.ALL_IN.value])
        except Exception:
            # best-effort fallback
            can_call = any(lm)
            can_check = False
            can_raise_3bb = False
            can_all_in = True if lm else False

        features[i] = 1.0 if can_call else 0.0; i += 1
        features[i] = 1.0 if can_check else 0.0; i += 1
        features[i] = 1.0 if can_raise_3bb else 0.0; i += 1
        features[i] = 1.0 if can_all_in else 0.0; i += 1

        # 49. number of legal moves normalized
        features[i] = clip01(len([x for x in lm if x]) / max(1, len(lm))); i += 1

        # 50. availability of any raise option (heuristic)
        features[i] = 1.0 if can_raise_3bb or can_all_in else 0.0; i += 1

        # 51. afford_allin_by_stack flag (1 if my_stack > 0)
        features[i] = 1.0 if my_stack > 0 else 0.0; i += 1

        # ------------------------
        # H. Derived ratios & misc (12)
        # ------------------------
        # 52. pot normalized (community + round) relative to big blind baseline
        features[i] = clip01((community_pot + current_round_pot) / (max(1.0, big_blind) * 20.0)); i += 1

        # 53. current_round_pot normalized by big blind
        features[i] = clip01(current_round_pot / (max(1.0, big_blind) * 20.0)); i += 1

        # 54. community_pot normalized by big blind
        features[i] = clip01(community_pot / (max(1.0, big_blind) * 20.0)); i += 1

        # 55. my stack as fraction of total stacks
        features[i] = clip01(my_stack / (sum(safe_float(s) for s in stacks) + 1e-9)); i += 1

        # 56. fraction of pot I currently own (my_contribution / total_pot)
        features[i] = clip01(my_contribution / (total_pot + 1e-9)); i += 1

        # 57. normalized difference: (equity - pot_odds)
        pot_odds = cost_to_call / (total_pot + cost_to_call + 1e-9)
        features[i] = clipm1p1((equity_alive - pot_odds)); i += 1

        # 58. "pressure" indicator : how big current_round_pot relative to average player stack
        avg_stack_safe = max(1e-9, avg_stack)
        features[i] = clip01(current_round_pot / (avg_stack_safe + 1e-9)); i += 1

        # 59. normalized min_call at my action if available
        min_call_list = None
        if isinstance(stage_data_list, (list, tuple)) and len(stage_data_list) > 0:
            try:
                min_call_list = stage_data_list[-1].get("min_call_at_action", None)
            except Exception:
                min_call_list = None
        min_call_here = safe_float(safe_list_get(min_call_list, position, 0.0)) if min_call_list is not None else 0.0
        features[i] = clip01(min_call_here / (max(1.0, big_blind) * 20.0)); i += 1

        # 60. fraction of stages (entries) where raises appeared (aggression_history)
        raise_entries = 0
        for sd in stage_data_list:
            try:
                raise_entries += 1 if any(sd.get("raises", [])) else 0
            except Exception:
                pass
        features[i] = clip01(raise_entries / max(1, len(stage_data_list))); i += 1

        # 61. fraction of stages where calls appeared
        call_entries = 0
        for sd in stage_data_list:
            try:
                call_entries += 1 if any(sd.get("calls", [])) else 0
            except Exception:
                pass
        features[i] = clip01(call_entries / max(1, len(stage_data_list))); i += 1

        # 62. tendency to have high min_call historically (avg min_call / big_blind baseline)
        historical_min_calls = []
        for sd in stage_data_list:
            try:
                mlist = sd.get("min_call_at_action", [])
                historical_min_calls.append(np.mean([safe_float(x) for x in mlist]) if mlist else 0.0)
            except Exception:
                historical_min_calls.append(0.0)
        avg_min_call_hist = np.mean(historical_min_calls) if historical_min_calls else 0.0
        features[i] = clip01(avg_min_call_hist / (max(1.0, big_blind) * 20.0)); i += 1

        # 63. simple hand-agnostic "conservative" flag: true if many players still active and my stack is small
        features[i] = 1.0 if (active_count > max(2, num_players // 2) and my_stack < avg_stack) else 0.0; i += 1

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