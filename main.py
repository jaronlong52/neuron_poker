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
        """
        Manual training loop for poker agent - similar to blackjack training
        This gives you full control over the training process
        """
        from agents.agent_my_agent import Player as MyAgent
        from agents.agent_random import Player as RandomPlayer
        
        print("="*60)
        print("MANUAL TRAINING MODE - Poker Agent")
        print("="*60)
        
        # Training configuration
        USE_TRAINED_WEIGHTS = False
        env_name = 'neuron_poker-v0'
        
        # Hyperparameters
        episodes = 50000  # Number of hands to play
        epsilon_start = 1.0 if not USE_TRAINED_WEIGHTS else 0.1
        epsilon_end = 0.01
        epsilon_decay = 0.99995  # Slightly faster decay for poker
        alpha = 0.01
        gamma = 0.99
        weights_file = "poker_weights.npy"
        
        print(f"Training Configuration:")
        print(f"  Episodes: {episodes}")
        print(f"  Initial epsilon: {epsilon_start}")
        print(f"  Final epsilon: {epsilon_end}")
        print(f"  Learning rate (alpha): {alpha}")
        print(f"  Discount factor (gamma): {gamma}")
        print(f"  Stack size: {self.stack}")
        print()
        
        # Create environment
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        
        # Create training agent with ALL required parameters
        training_agent = MyAgent(
            epsilon=epsilon_start,
            alpha=alpha,
            gamma=gamma,
            name="TrainingAgent",
            stack_size=self.stack,
            weights_file=weights_file if USE_TRAINED_WEIGHTS else None
        )
        
        # Verify agent has all necessary attributes
        assert hasattr(training_agent, 'name'), "Agent missing 'name' attribute"
        assert hasattr(training_agent, 'stack'), "Agent missing 'stack' attribute"
        assert hasattr(training_agent, 'autoplay'), "Agent missing 'autoplay' attribute"
        assert hasattr(training_agent, 'epsilon'), "Agent missing 'epsilon' attribute"
        assert hasattr(training_agent, 'alpha'), "Agent missing 'alpha' attribute"
        assert hasattr(training_agent, 'gamma'), "Agent missing 'gamma' attribute"
        assert hasattr(training_agent, 'weights'), "Agent missing 'weights' attribute"
        
        # Add players to environment
        self.env.unwrapped.add_player(training_agent)
        for i in range(5):
            self.env.unwrapped.add_player(RandomPlayer(name=f'Random_{i+1}'))
        
        # Training tracking
        episode_rewards = []
        wins = []
        agent_index = 0  # Training agent is first player
        
        # Storage for experience tuples during an episode
        episode_experiences = []
        
        print("Starting training...")
        print()
        
        for episode in range(episodes):
            # Reset environment for new hand
            observation, info = self.env.reset()
            
            # Clear experiences from previous episode
            episode_experiences = []
            
            # Track episode data
            episode_reward = 0
            done = False
            step_count = 0
            
            # Play one complete hand (episode)
            while not done:
                # Get current player index
                current_player_idx = self.env.unwrapped.current_player_seat
                
                # Check if it's our training agent's turn
                if current_player_idx == agent_index:
                    # Extract the processed observation that agent uses
                    prev_info_obs = info.get('observation', {})
                    
                    # Get valid actions for current state
                    action_space = self.env.unwrapped.legal_moves
                    
                    # Agent chooses action (epsilon-greedy handled in agent.action())
                    action = training_agent.action(action_space, observation, info)
                    
                    # Execute action in environment
                    next_observation, reward, terminated, truncated, next_info = self.env.step(action)
                    done = terminated or truncated
                    
                    # Extract next processed observation
                    next_info_obs = next_info.get('observation', {})
                    
                    # Store experience tuple with the PROCESSED observations
                    episode_experiences.append({
                        'obs': prev_info_obs,           # Processed observation (dict)
                        'action': action,               # Action enum
                        'reward': reward,               # Reward received
                        'next_obs': next_info_obs,      # Next processed observation (dict)
                        'done': done                    # Terminal flag
                    })
                    
                    # Update for next iteration
                    observation = next_observation
                    info = next_info
                    episode_reward += reward
                    
                else:
                    # Other player's turn - just step the environment
                    next_observation, reward, terminated, truncated, next_info = self.env.step(None)
                    done = terminated or truncated
                    observation = next_observation
                    info = next_info
                
                step_count += 1
                
                # Safety check to prevent infinite loops
                if step_count > 1000:
                    print(f"Warning: Episode {episode} exceeded 1000 steps. Breaking.")
                    done = True
            
            # Episode finished - now do the learning updates
            for exp in episode_experiences:
                # Call agent's update method with processed observations
                training_agent.update(
                    obs=exp['obs'],           # Dict from info['observation']
                    action=exp['action'],     # Action enum
                    reward=exp['reward'],     # Float reward
                    next_obs=exp['next_obs'], # Dict from next info['observation']
                    terminated=exp['done']    # Boolean
                )
            
            # Track episode statistics
            episode_rewards.append(episode_reward)
            winner_idx = self.env.unwrapped.winner_ix
            wins.append(1 if winner_idx == agent_index else 0)
            
            # Decay epsilon
            training_agent.epsilon = max(epsilon_end, training_agent.epsilon * epsilon_decay)
            
            # Print progress
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:])
                win_rate = np.mean(wins[-1000:]) * 100
                avg_error = np.mean(training_agent.training_error[-1000:]) if training_agent.training_error else 0
                
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  Avg Reward (last 1000): {avg_reward:.3f}")
                print(f"  Win Rate (last 1000): {win_rate:.1f}%")
                print(f"  Epsilon: {training_agent.epsilon:.4f}")
                print(f"  Avg TD Error: {avg_error:.4f}")
                print()
            
            # Save weights periodically
            if (episode + 1) % 10000 == 0:
                checkpoint_file = f"poker_weights_ep{episode + 1}.npy"
                training_agent.save_weights(checkpoint_file)
                print(f"Checkpoint saved: {checkpoint_file}")
        
        # Training complete
        print("="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Final Statistics (last 1000 episodes):")
        print(f"  Average Reward: {np.mean(episode_rewards[-1000:]):.3f}")
        print(f"  Win Rate: {np.mean(wins[-1000:]) * 100:.1f}%")
        print(f"  Final Epsilon: {training_agent.epsilon:.4f}")
        print()
        
        # Save final weights
        final_weights_file = "poker_weights_final.npy"
        training_agent.save_weights(final_weights_file)
        print(f"Final weights saved to: {final_weights_file}")
        
        # Plot training curves (optional)
        if self.funds_plot:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot rewards
            window = 100
            rewards_smooth = pd.Series(episode_rewards).rolling(window=window).mean()
            axes[0].plot(rewards_smooth)
            axes[0].set_title(f'Average Reward (smoothed over {window} episodes)')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True)
            
            # Plot win rate
            win_rate_smooth = pd.Series(wins).rolling(window=window).mean() * 100
            axes[1].plot(win_rate_smooth)
            axes[1].set_title(f'Win Rate % (smoothed over {window} episodes)')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Win Rate %')
            axes[1].axhline(y=16.67, color='r', linestyle='--', label='Random (1/6 players)')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig('poker_training_curves.png')
            print("Training curves saved to: poker_training_curves.png")
            plt.show()

    def my_agent_evaluation(self):
        """
        Evaluation mode - test trained agent without exploration
        """
        from agents.agent_my_agent import Player as MyAgent
        from agents.agent_random import Player as RandomPlayer
        from agents.agent_consider_equity import Player as EquityPlayer
        print("="*60)
        print("EVALUATION MODE - Testing Trained Agent")
        print("="*60)
        env_name = 'neuron_poker-v0'
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        # Load trained agent with epsilon=0 (no exploration)
        trained_agent = MyAgent(
            epsilon=0.0,  # Pure exploitation
            alpha=0.0,    # No learning
            gamma=0.99,
            name="TrainedAgent",
            stack_size=self.stack,
            weights_file="poker_weights_final.npy"
        )
        # Add diverse opponents
        self.env.unwrapped.add_player(trained_agent)
        self.env.unwrapped.add_player(EquityPlayer(name='Equity_Tight', min_call_equity=.6, min_bet_equity=.7))
        self.env.unwrapped.add_player(EquityPlayer(name='Equity_Loose', min_call_equity=.3, min_bet_equity=.4))
        for i in range(3):
            self.env.unwrapped.add_player(RandomPlayer(name=f'Random_{i+1}'))
        # Run evaluation games
        eval_episodes = self.num_episodes if self.num_episodes > 100 else 1000
        wins = []
        rewards = []
        print(f"Running {eval_episodes} evaluation episodes...")
        print()
        for episode in range(eval_episodes):
            self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                _, reward, terminated, truncated, _ = self.env.step(None)
                done = terminated or truncated
                if self.env.unwrapped.current_player_seat == 0:
                    episode_reward += reward
            winner_idx = self.env.unwrapped.winner_ix
            wins.append(1 if winner_idx == 0 else 0)
            rewards.append(episode_reward)
            if (episode + 1) % 100 == 0:
                print(f"Progress: {episode + 1}/{eval_episodes} - "
                      f"Current Win Rate: {np.mean(wins) * 100:.1f}%")
        # Print results
        print()
        print("="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Episodes: {eval_episodes}")
        print(f"Win Rate: {np.mean(wins) * 100:.1f}%")
        print(f"Average Reward: {np.mean(rewards):.3f}")
        print(f"Expected Random Win Rate: {100/6:.1f}% (1 out of 6 players)")
        print()
        if np.mean(wins) * 100 > 100/6:
            print("✓ Agent is performing BETTER than random!")
        else:
            print("✗ Agent needs more training or hyperparameter tuning")


if __name__ == '__main__':
    command_line_parser()
