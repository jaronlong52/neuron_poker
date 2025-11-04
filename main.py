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
  main.py selfplay my-agent-train [options]

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

"""

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
    print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

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
            runner.my_agent_play()
        
        elif args['my-agent-train']:
            runner.my_agent_manual_training()
            
        elif args['my-agent-eval']:
            runner.my_agent_evaluation()


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




    def my_agent_manual_training(self):
        from agents.agent_my_agent import Player as MyAgent
        from agents.agent_random import Player as RandomPlayer
        from gym_env.env import HoldemTable
        from gym_env.cycle import PlayerCycle

        print("="*60)
        print("FINAL MANUAL TRAINING - Stable Q-Learning Poker Agent")
        print("="*60)

        num_games_to_play = self.num_episodes
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.9999
        alpha = 0.005
        gamma = 0.95
        dest_weights_file = "poker_weights_final.npy"
        source_weights_file = None

        print(f"Target Games: {num_games_to_play}, Alpha: {alpha}, Gamma: {gamma}")

        # -------------------------------------------------
        # 1. Environment Setup
        # -------------------------------------------------
        env_args = {
            'initial_stacks': self.stack,
            'small_blind': 10,
            'big_blind': 20,
            'render': self.render,
            'funds_plot': self.funds_plot,
            'use_cpp_montecarlo': self.use_cpp_montecarlo,
            'calculate_equity': True
        }
        self.env = HoldemTable(**env_args)

        # -------------------------------------------------
        # 2. Create Training Agent
        # -------------------------------------------------
        training_agent = MyAgent(
            epsilon=epsilon_start,
            alpha=alpha,
            gamma=gamma,
            name="QAgent",
            stack_size=self.stack,
            weights_file=source_weights_file
        )

        self.env.add_player(training_agent)
        for i in range(2):
            random_agent = RandomPlayer(name=f'Random_{i+1}')
            random_agent.autoplay = True
            self.env.add_player(random_agent)

        # -------------------------------------------------
        # 3. Training Loop
        # -------------------------------------------------
        episode_rewards = []
        wins = []
        agent_seat = 0
        games_played = 0
        hands_played = 0

        print("Training started...")

        while games_played < num_games_to_play:
            self.log.info(f"--- STARTING GAME {games_played + 1}/{num_games_to_play} ---")

            # Reset stacks for new game
            for player in self.env.players:
                player.stack = self.stack

            # Reinitialize player cycle
            max_steps_after_raiser = (self.env.max_raises_per_player_round - 1) * len(self.env.players) - 1
            self.env.player_cycle = PlayerCycle(
                self.env.players,
                dealer_idx=-1,
                max_steps_after_raiser=max_steps_after_raiser,
                max_steps_after_big_blind=len(self.env.players),
                max_raises_per_player_round=self.env.max_raises_per_player_round
            )
            self.env.done = False

            # Play hands until someone goes bust
            while not self.env.done:
                # Check if we can still play
                players_alive = sum(1 for p in self.env.players if p.stack > 0)
                if players_alive < 2:
                    break

                obs_array, info = self.env.reset()

                if self.env.done:
                    break

                hand_reward = 0
                hand_complete = False

                while not hand_complete and not self.env.done:
                    if self.env.current_player is None:
                        hand_complete = True
                        break

                    current_seat = self.env.current_player.seat
                    current_obs = self.env.observation
                    current_info = self.env.info
                    legal_moves = self.env.legal_moves

                    if not legal_moves:
                        hand_complete = True
                        break

                    if current_seat == agent_seat:
                        action = training_agent.action(legal_moves, current_obs, current_info)
                        next_obs, reward, done, truncated, next_info = self.env.step(action)

                        training_agent.update(
                            current_obs, current_info, action, reward,
                            next_obs, next_info, done
                        )

                        hand_reward += reward

                        if done or truncated or self.env.done:
                            hand_complete = True
                    else:
                        next_obs, reward, done, truncated, next_info = self.env.step(None)

                        if done or truncated or self.env.done:
                            hand_complete = True

                # Record hand stats
                hands_played += 1
                episode_rewards.append(hand_reward)

                if self.env.winner_ix is not None:
                    wins.append(1 if self.env.winner_ix == agent_seat else 0)
                else:
                    wins.append(0)

                training_agent.epsilon = max(epsilon_end, training_agent.epsilon * epsilon_decay)

            games_played += 1
            self.log.info(f"--- GAME {games_played}/{num_games_to_play} COMPLETE ---")
            self.log.info(f"Total hands played: {hands_played}")

        # Save and finish
        training_agent.save_weights(dest_weights_file)
        print(f"FINAL WEIGHTS: {dest_weights_file}")
        print(f"Training complete. Played {games_played} games across {hands_played} hands.")

        if self.funds_plot and len(episode_rewards) > 10:
            import matplotlib.pyplot as plt
            window = min(max(10, len(episode_rewards) // 10), 100)
            r_smooth = pd.Series(episode_rewards).rolling(window).mean()
            w_smooth = pd.Series(wins).rolling(window).mean() * 100

            plt.figure(figsize=(12,8))
            plt.subplot(2,1,1)
            plt.plot(r_smooth)
            plt.title(f'Average Reward ({window}-hand window)')
            plt.ylabel('Reward')
            plt.grid()

            plt.subplot(2,1,2)
            plt.plot(w_smooth)
            plt.axhline(33.33, color='r', ls='--', label='Random (33%)')
            plt.title('Win Rate %')
            plt.ylabel('Win %')
            plt.xlabel('Hand Number')
            plt.legend()
            plt.tight_layout()
            plt.savefig('poker_final_curves.png')
            plt.show()


    def my_agent_evaluation(self):
        from agents.agent_my_agent import Player as MyAgent
        from agents.agent_random import Player as RandomPlayer
        from agents.agent_consider_equity import Player as EquityPlayer
        from gym_env.env import HoldemTable

        print("="*60)
        print("EVALUATION MODE - Testing Trained Agent")
        print("="*60)

        # -------------------------------------------------
        # 1. Config + env
        # -------------------------------------------------
        # Environment
        env_args = {
            'initial_stacks': self.stack,
            'small_blind': 10,
            'big_blind': 20,
            'render': self.render,
            'funds_plot': self.funds_plot,
            'use_cpp_montecarlo': self.use_cpp_montecarlo,
            'calculate_equity': True  # MyAgent needs equity to extract features
        }
        self.env = HoldemTable(**env_args)

        # -------------------------------------------------
        # 2. Add agents
        # -------------------------------------------------
        training_agent = MyAgent(
            epsilon=0.0, alpha=0.0, gamma=0.99,
            name="Trained", stack_size=self.stack,
            weights_file="poker_weights_final.npy"
        )
        self.env.add_player(training_agent)                                 # seat 0
        self.env.add_player(EquityPlayer(name='Tight',  min_call_equity=.6, min_bet_equity=.7))
        self.env.add_player(EquityPlayer(name='Loose',  min_call_equity=.3, min_bet_equity=.4))
        for i in range(3):
            self.env.add_player(RandomPlayer(name=f'Rand_{i+1}'))

        # -------------------------------------------------
        # 3. Evaluation loop
        # -------------------------------------------------
        eval_episodes = max(self.num_episodes, 1000)
        wins, rewards = [], []

        print(f"Running {eval_episodes} evaluation episodes...\n")

        for ep in range(eval_episodes):
            obs_array, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                seat = self.env.current_player.seat
                legal = self.env._get_legal_moves()
            
                if seat == 0:
                    # 1. Cache the current state
                    current_obs = obs_array
                    
                    # 2. Get action
                    action = training_agent.action(legal, current_obs, info)
                    
                    # 3. Take step (and get next_info)
                    next_obs, reward, terminated, truncated, next_info = self.env.step(action) # Don't discard info with '_'
                    
                    # 4. Call update with all arguments
                    training_agent.update(current_obs, action, reward, next_obs, terminated) 
                    
                    episode_reward += reward
                else:
                    # ... (opponent logic)
                    opp = self.env.players[seat]
                    opp_action = opp.agent_obj.action(legal, current_obs, info)
                    next_obs, reward, terminated, truncated, next_info = self.env.step(opp_action) # Update next_info
            
                done = terminated or truncated
                obs_array = next_obs
                info = next_info # Pass the new info dict to the next loop iteration

            wins.append(1 if self.env.winner_ix == 0 else 0)
            rewards.append(episode_reward)

            if (ep + 1) % 100 == 0:
                print(f"Eval {ep+1}/{eval_episodes} – Win% {np.mean(wins)*100:.1f}")

        # -------------------------------------------------
        # 4. Results
        # -------------------------------------------------
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Win rate: {np.mean(wins)*100:.2f}% (random: 16.67%)")
        print(f"Avg reward: {np.mean(rewards):.2f}")
        if np.mean(wins) > 1/6:
            print("Agent BEATS random!")
        else:
            print("Agent needs more training.")


if __name__ == '__main__':
    command_line_parser()
