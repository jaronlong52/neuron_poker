from gym_env.env import Action

class Player:
    def __init__(self, name="MyAgent", stack_size=None):
        self.name = name
        self.stack = stack_size
        self.autoplay = True

    def action(self, action_space, observation, info):
        obs = info.get('observation', {})
        valid = [a for a in action_space if a.value != 0]

        # === GET EQUITY SAFELY ===
        equity = 0.0
        eq = obs.get('equity')
        if eq and isinstance(eq, dict):
            equity = eq.get('me', 0.0)

        stage = obs.get('stage', '')

        # === PRE-FLOP: Tight ===
        if stage == 'PREFLOP':
            call_cost = obs.get('current_bet', 0) - obs.get('my_contribution', 0)
            if Action.CALL in valid and call_cost <= 2:
                return Action.CALL
            if Action.FOLD in valid:
                return Action.FOLD
            if Action.CHECK in valid:
                return Action.CHECK
            return valid[0]

        # === POST-FLOP: Use valid raise actions ===
        # Replace Action.RAISE with any of these:
        if Action.RAISE_POT in valid and equity > 0.65:
            return Action.RAISE_POT
        if Action.RAISE_HALF_POT in valid and equity > 0.60:
            return Action.RAISE_HALF_POT
        if Action.RAISE_3BB in valid and equity > 0.55:
            return Action.RAISE_3BB

        if Action.CALL in valid and equity > 0.35:
            return Action.CALL

        # === CHECK if FOLD not allowed ===
        if Action.CHECK in valid:
            return Action.CHECK
        if Action.FOLD in valid:
            return Action.FOLD

        # === SAFETY: Always return valid ===
        return valid[0] if valid else Action.FOLD

    def __str__(self):
        return self.name