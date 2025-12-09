"""
neuron poker

Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
  main.py learn_table_scraping [options]
  main.py selfplay my-agent [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500].
  --test                    test mode for my-agent
  --weights_file=<>         weights file for my-agent

"""

from datetime import datetime
import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from docopt import docopt

from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


# pylint: disable=import-outside-toplevel

def command_line_parser():
    """Entry function"""
    args = docopt(__doc__)
    if args['--log']:
        logfile = args['--log']
    else:
        print("Using default log file")
        logfile = 'default'
    model_name = args['--name'] if args['--name'] else 'dqn1'
    screenloglevel = logging.INFO if not args['--screenloglevel'] else \
        getattr(logging, args['--screenloglevel'].upper())
    _ = get_config()
    init_logger(screenlevel=screenloglevel, filename=logfile)
    # print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    # log.info("Initializing program")

    if args['selfplay']:
        num_episodes = 1 if not args['--episodes'] else int(args['--episodes'])
        runner = SelfPlay(render=args['--render'], num_episodes=num_episodes,
                          use_cpp_montecarlo=args['--use_cpp_montecarlo'],
                          funds_plot=args['--funds_plot'],
                          stack=int(args['--stack']))

        if args['random']:
            runner.random_agents()

        elif args['keypress']:
            runner.key_press_agents()

        elif args['consider_equity']:
            runner.equity_vs_random()

        elif args['equity_improvement']:
            improvement_rounds = int(args['--improvement_rounds'])
            runner.equity_self_improvement(improvement_rounds)

        elif args['dqn_train']:
            runner.dqn_train_keras_rl(model_name)

        elif args['dqn_play']:
            runner.dqn_play_keras_rl(model_name)

        elif args['my-agent']:
            test_mode = False if not args['--test'] else True
            weights_file = None if not args['--weights_file'] else args['--weights_file']
            runner.my_agent_play(test_mode, weights_file)


    else:
        raise RuntimeError("Argument not yet implemented")


class SelfPlay:
    """Orchestration of playing against itself"""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=500):
        """Initialize"""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack
        self.log = logging.getLogger(__name__)

    def random_agents(self):
        """Create an environment with 6 random players"""
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        for _ in range(num_of_plrs):
            player = RandomPlayer()
            self.env.unwrapped.add_player(player)  # Use unwrapped to access add_player
    
        self.env.reset()

    def key_press_agents(self):
        """Create an environment with 2 key-press agents (human-controlled)"""
        from agents.agent_keypress import Player as KeyPressAgent
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2                     # change if you want more opponents
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)

        for _ in range(num_of_plrs):
            player = KeyPressAgent()
            self.env.unwrapped.add_player(player)   # <-- UNWRAPPED!

        self.env.reset()

    def equity_vs_random(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def equity_self_improvement(self, improvement_rounds):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        calling = [.1, .2, .3, .4, .5, .6]
        betting = [.2, .3, .4, .5, .6, .7]

        for improvement_round in range(improvement_rounds):
            env_name = 'neuron_poker-v0'
            self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
            for i in range(6):
                self.env.add_player(EquityPlayer(name=f'Equity/{calling[i]}/{betting[i]}',
                                                 min_call_equity=calling[i],
                                                 min_bet_equity=betting[i]))

            for _ in range(self.num_episodes):
                self.env.reset()
                self.winner_in_episodes.append(self.env.winner_ix)

            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = int(league_table.index[0])
            print(league_table)
            print(f"Best Player: {best_player}")

            # self improve:
            self.log.info(f"Self improvment round {improvement_round}")
            for i in range(6):
                calling[i] = np.mean([calling[i], calling[best_player]])
                self.log.info(f"New calling for player {i} is {calling[i]}")
                betting[i] = np.mean([betting[i], betting[best_player]])
                self.log.info(f"New betting for player {i} is {betting[i]}")

    def dqn_train_keras_rl(self, model_name):
        """Implementation of kreras-rl deep q learing."""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        env = gym.make(env_name, initial_stacks=self.stack, funds_plot=self.funds_plot, render=self.render,
                       use_cpp_montecarlo=self.use_cpp_montecarlo)

        np.random.seed(123)
        env.seed(123)
        env.add_player(EquityPlayer(name='equity/50/70', min_call_equity=.5, min_bet_equity=.7))
        env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        env.add_player(RandomPlayer())
        env.add_player(RandomPlayer())
        env.add_player(RandomPlayer())
        env.add_player(PlayerShell(name='keras-rl', stack_size=self.stack))  # shell is used for callback to keras rl

        env.reset()

        dqn = DQNPlayer()
        dqn.initiate_agent(env)
        dqn.train(env_name=model_name)

    def dqn_play_keras_rl(self, model_name):
        """Create 6 players, one of them a trained DQN"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(PlayerShell(name='keras-rl', stack_size=self.stack))

        self.env.reset()

        dqn = DQNPlayer(load_model=model_name, env=self.env)
        dqn.play(nb_episodes=self.num_episodes, render=self.render)

    def dqn_train_custom_q1(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_custom_q1 import Player as Custom_Q1
        from agents.agent_random import Player as RandomPlayer
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        # self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        # self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        # self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        # self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(Custom_Q1(name='Deep_Q1'))

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")


    def my_agent_test(self, agent, weights_path, save_file="temp_name", test_episodes=100):
        """
        Test the agent using loaded weights against opponents.
        Tracks win counts and visualizes results.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime

        # Configure agent for testing (no learning, no exploration)
        agent.load_weights(weights_path)
        agent.isNotLearning = True
        agent.epsilon = 0.0

        # Initialize tracking
        player_names = [p.name for p in self.env.unwrapped.players]
        win_counts = {name: 0 for name in player_names}

        print("=" * 60)
        print(f"TESTING AGENT: {test_episodes} episodes")
        print(f"Agent: {agent.name}")
        print(f"Opponents: {', '.join([n for n in player_names if n != agent.name])}")
        print("=" * 60)

        start_time = datetime.now()

        # Run test episodes
        for ep in range(test_episodes):
            print(f"Progress: {ep + 1}/{test_episodes} episodes completed")

            # Run full episode
            obs, info = self.env.reset()

            # Get final stacks from the info dictionary
            # The stacks are normalized by (BB * 100), so multiply back
            if info and 'player_data' in info:
                final_stacks = np.array(info['player_data']['stack']) * (self.big_blind * 100)
            else:
                # Fallback: try to get from environment's funds_history
                env_unwrapped = self.env.unwrapped
                if hasattr(env_unwrapped, 'funds_history') and len(env_unwrapped.funds_history) > 0:
                    final_stacks = env_unwrapped.funds_history.iloc[-1].values
                else:
                    print(f"[WARNING] Episode {ep + 1}: No final stack data available")
                    continue
                
            # Determine winner (player with highest stack)
            winner_idx = np.argmax(final_stacks)
            winner_name = player_names[winner_idx]
            win_counts[winner_name] += 1

            # Clear agent history for next episode
            agent.episode_history = []

        end_time = datetime.now()
        duration = end_time - start_time

        # ========== RESULTS SUMMARY ==========
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Average time per episode: {duration.total_seconds() / test_episodes:.2f}s")
        print(f"\nWin Distribution:")

        for name in player_names:
            win_pct = (win_counts[name] / test_episodes) * 100
            print(f"  {name:20s}: {win_counts[name]:4d} wins ({win_pct:5.1f}%)")

        print("=" * 60)

        # ========== VISUALIZATION ==========
        fig, ax = plt.subplots(figsize=(10, 6))

        # Win Counts Bar Chart
        bars = ax.bar(win_counts.keys(), win_counts.values(), color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_ylabel("Number of Wins", fontsize=11)
        ax.set_title(f"Win Distribution - {agent.name} vs Opponents ({test_episodes} Games)", 
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)

        # Add win count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Adjust y-limit for label space
        max_wins = max(win_counts.values()) if win_counts.values() else 0
        ax.set_ylim(0, max_wins * 1.15)

        plt.tight_layout()

        # Save plot
        save_file += "_test_results.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_file}")
        plt.show()


    def my_agent_play(self, test_mode=False, weights_file=None):
        """1 MyAgent vs 5 random"""
        from agents.agent_my_agent import Player as MyAgent
        from agents.agent_random import Player as RandomPlayer
        from agents.agent_consider_equity import Player as EquityPlayer

        env_name = 'neuron_poker-v0'

        self.stack = 100 # hard coded for simplicity
        self.big_blind = 2
        num_episodes = 2000
        num_passes = 2
        e_decay = 0.9975

        # Name of the file to save weights to after training
        save_weights_to_file = "weights_s100_bb2_epi2000_passes2_featuresV7"
        # Name of the file to save plotted results to after training
        save_plot_to_file = "td_error_s100_bb2_epi2000_passes2_featuresV7"

        # Name of file to load weights of last trained agent from
        load_weights_last_trained_model = "weights_s100_bb2_epi2000_passes2_featuresV7"

        training_agent = MyAgent(
            epsilon=1.0,
            epsilon_decay=e_decay,
            alpha=0.005,
            gamma=0.95,
            big_blind=self.big_blind,
            name="QAgent",
            stack_size=self.stack,
            num_update_passes=num_passes,
            isNotLearning=False,
            num_episodes=num_episodes # Only for Graphing Purposes
        )

        self.env = gym.make(env_name, initial_stacks=self.stack, small_blind=1, big_blind=self.big_blind, render=self.render, funds_plot=False)

        # Add players via unwrapped
        last_trained_model = MyAgent(
            epsilon=0.0,
            epsilon_decay=0.0,
            alpha=0.0,
            gamma=0.0,
            big_blind=self.big_blind,
            name="LastTrainedModel_1",
            stack_size=self.stack,
            weights_file=load_weights_last_trained_model,
            isNotLearning=True
        )
        last_trained_model_2 = MyAgent(
            epsilon=0.0,
            epsilon_decay=0.0,
            alpha=0.0,
            gamma=0.0,
            big_blind=self.big_blind,
            name="LastTrainedModel_2",
            stack_size=self.stack,
            weights_file=load_weights_last_trained_model,
            isNotLearning=True
        )
        last_trained_model_3 = MyAgent(
            epsilon=0.0,
            epsilon_decay=0.0,
            alpha=0.0,
            gamma=0.0,
            big_blind=self.big_blind,
            name="LastTrainedModel_3",
            stack_size=self.stack,
            weights_file=load_weights_last_trained_model,
            isNotLearning=True
        )

        # *** TRAINING AGENT MUST BE ADDED FIRST ***
        # Due to added early termination condition in _check_game_over()
        self.env.unwrapped.add_player(training_agent)
        
        self.env.unwrapped.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        self.env.unwrapped.add_player(RandomPlayer(name=f'Random_1'))
        self.env.unwrapped.add_player(last_trained_model_2)
        self.env.unwrapped.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        self.env.unwrapped.add_player(last_trained_model_3)
        self.env.unwrapped.add_player(RandomPlayer(name=f'Random_2'))
        self.env.unwrapped.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.unwrapped.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        self.env.unwrapped.add_player(last_trained_model)

        # -------------------------
        # *** TEST MODE BRANCH ***
        # -------------------------
        if test_mode:
            if weights_file is None:
                raise ValueError("weights_file must be provided when test_mode=True.")

            self.my_agent_test(
                agent=training_agent,
                weights_path=weights_file,
                save_file=save_plot_to_file
            )
            return  # prevent running training afterwards


        # -------------------------
        # *** TRAINING MODE ***
        # -------------------------
        print("$$ Starting training...")
        start_time = datetime.now()

        for ep in range(num_episodes):
            print(f"\n$$ Episode {ep + 1}/{num_episodes}")
            self.env.reset()
            training_agent.decay_epsilon()
            # print("---------------------------")
            # print("Num Actions: ", training_agent.num_actions)
            # print("Num Updates: ", training_agent.num_updates)
            # print("Num markers: ", training_agent.num_markers)
            # print("---------------------------")
            training_agent.num_actions = 0
            training_agent.num_updates = 0
            training_agent.num_markers = 0
            training_agent.episode_history = []

        print("$$ Training finished.")
        end_time = datetime.now()

        time_difference = end_time - start_time
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")
        print(f"Duration: {time_difference}")
        print(f"Agent Wins: {training_agent.game_wins} out of {num_episodes} games.")

        training_agent.save_weights(save_weights_to_file)

        # After game finishes, show training results
        training_agent.plot_td_error(save_plot_to_file)


if __name__ == '__main__':
    command_line_parser()
