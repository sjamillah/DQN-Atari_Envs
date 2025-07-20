import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py
import time
import matplotlib.pyplot as plt
import cv2
import json


class LocalDQNPlayer:
    """Play with trained DQN agent in local environment with GUI display."""
    
    def __init__(self, env_name="ALE/Boxing-v5", model_path="dqn_model.zip", 
                 num_episodes=5, delay_between_actions=0.05, save_video=False):
        """
        Initialize the local DQN player.
        
        Args:
            env_name: Name of the Atari environment
            model_path: Path to the trained DQN model
            num_episodes: Number of episodes to play
            delay_between_actions: Delay in seconds between actions
            save_video: Whether to save gameplay as video file
        """
        self.env_name = env_name
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.delay_between_actions = delay_between_actions
        self.save_video = save_video
        
        self.env = None
        self.model = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.video_writer = None
        
        print(f"Local DQN Player for {self.env_name}")
        print(f"Model: {self.model_path}")
        print(f"Episodes: {self.num_episodes}")

    def find_model_file(self):
        """Search for the model file in common locations."""
        if os.path.exists(self.model_path):
            print(f"Found model at: {self.model_path}")
            return self.model_path
        
        print(f"Model not found at {self.model_path}, searching alternatives...")
        
        search_paths = [
            f"models/final/dqn_{self.env_name.replace('/', '_')}_CnnPolicy.zip",
            "models/final/dqn_model.zip",
            "models/dqn_model.zip",
            "dqn_model.zip",
            f"saved_models/dqn_{self.env_name.replace('/', '_')}.zip"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                self.model_path = path
                return path
        
        raise FileNotFoundError(
            f"Could not find trained model. Searched:\n" +
            f"- {self.model_path}\n" +
            "\n".join(f"- {p}" for p in search_paths) +
            "\n\nPlease ensure the model file exists."
        )

    def setup_environment(self, render_mode='human', seed=42):
        """Create the Atari environment for local display."""
        print(f"\nSetting up {self.env_name} environment...")
        
        try:
            self.env = make_atari_env(
                self.env_name, 
                n_envs=1, 
                seed=seed,
                env_kwargs={'render_mode': render_mode}
            )
            self.env = VecFrameStack(self.env, n_stack=4)
            
            print("Environment ready")
            print(f"Observation space: {self.env.observation_space}")
            print(f"Action space: {self.env.action_space}")
            
        except Exception as e:
            raise RuntimeError(f"Environment setup failed: {e}")

    def load_trained_model(self):
        """Load the DQN model."""
        model_path = self.find_model_file()
        
        print(f"\nLoading model from {model_path}...")
        
        try:
            self.model = DQN.load(model_path, env=self.env)
            print("Model loaded successfully")
            print(f"Policy: {self.model.policy}")
            print(f"Device: {self.model.device}")
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def setup_video_writer(self, frame):
        """Initialize video writer if video saving is enabled."""
        if not self.save_video:
            return
            
        video_filename = f"gameplay_{self.env_name.replace('/', '_')}.mp4"
        height, width = frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_filename, fourcc, 20.0, (width, height)
        )
        print(f"Recording video to {video_filename}")

    def save_frame_to_video(self, frame):
        """Save frame to video file if recording."""
        if self.video_writer is not None and frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

    def play_single_episode(self, episode_number):
        """Play one complete episode."""
        print(f"\nEpisode {episode_number}/{self.num_episodes}")
        
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        episode_done = False
        
        print("Episode starting...")
        
        while not episode_done:
            # Get action from trained model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, done, info = self.env.step(action)
            
            # Update episode statistics
            total_reward += reward[0]
            steps += 1
            
            # Get frame for video recording
            if self.save_video:
                frame = self.env.render()
                if frame is not None:
                    if self.video_writer is None:
                        self.setup_video_writer(frame)
                    self.save_frame_to_video(frame)
            
            # Add delay for better viewing
            time.sleep(self.delay_between_actions)
            
            if done[0]:
                episode_done = True
        
        print(f"Episode {episode_number} finished")
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        
        return total_reward, steps

    def play_all_episodes(self):
        """Play all configured episodes."""
        print(f"\nStarting {self.num_episodes} episodes")
        print("Press Ctrl+C to stop early")
        print("Close the game window to end current episode")
        
        self.episode_rewards = []
        self.episode_lengths = []
        
        try:
            for episode in range(1, self.num_episodes + 1):
                episode_reward, episode_length = self.play_single_episode(episode)
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                if episode < self.num_episodes:
                    print("Next episode in 2 seconds...")
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"\nError during gameplay: {e}")

    def calculate_performance_stats(self):
        """Calculate performance statistics from episodes."""
        if not self.episode_rewards:
            return {}
            
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'max_length': np.max(self.episode_lengths),
            'min_length': np.min(self.episode_lengths),
            'total_steps': np.sum(self.episode_lengths)
        }

    def display_results(self):
        """Display performance results."""
        stats = self.calculate_performance_stats()
        
        if not stats:
            print("No episodes completed")
            return
            
        print(f"\nPerformance Summary:")
        print(f"Episodes played: {stats['total_episodes']}")
        print(f"Average reward: {stats['mean_reward']:.2f}")
        print(f"Best reward: {stats['max_reward']:.2f}")
        print(f"Worst reward: {stats['min_reward']:.2f}")
        print(f"Reward std dev: {stats['std_reward']:.2f}")
        print(f"Average episode length: {stats['mean_length']:.1f} steps")
        print(f"Total steps: {stats['total_steps']}")
        
        print(f"\nDetailed results:")
        for i, (reward, length) in enumerate(zip(self.episode_rewards, self.episode_lengths), 1):
            print(f"  Episode {i}: {reward:.2f} reward, {length} steps")

    def create_results_plot(self):
        """Generate performance plots."""
        if not self.episode_rewards:
            print("No data to plot")
            return
            
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rewards plot
        ax1.plot(episodes, self.episode_rewards, 'b-o', linewidth=2, markersize=6)
        ax1.axhline(y=np.mean(self.episode_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode lengths plot
        ax2.plot(episodes, self.episode_lengths, 'g-o', linewidth=2, markersize=6)
        ax2.axhline(y=np.mean(self.episode_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(self.episode_lengths):.1f}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_filename = f"results_{self.env_name.replace('/', '_')}.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.show()

    def save_results_to_file(self, filename=None):
        """Save results to JSON file."""
        if filename is None:
            filename = f"results_{self.env_name.replace('/', '_')}.json"
            
        stats = self.calculate_performance_stats()
        if not stats:
            print("No results to save")
            return
            
        results_data = {
            'environment': self.env_name,
            'model_path': self.model_path,
            'configuration': {
                'num_episodes': self.num_episodes,
                'delay_between_actions': self.delay_between_actions
            },
            'statistics': stats,
            'episode_data': {
                'rewards': self.episode_rewards,
                'lengths': self.episode_lengths
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")

    def cleanup_resources(self):
        """Clean up environment and video resources."""
        if self.video_writer:
            self.video_writer.release()
            print("Video file saved")
            
        if self.env:
            self.env.close()
            print("Environment closed")

    def run_gameplay_session(self):
        """Execute complete gameplay session."""
        try:
            # Initialize everything
            self.setup_environment()
            self.load_trained_model()
            
            # Play episodes
            self.play_all_episodes()
            
            # Show results
            self.display_results()
            self.create_results_plot()
            self.save_results_to_file()
            
            print("\nGameplay session complete!")
            return self.calculate_performance_stats()
            
        except Exception as e:
            print(f"Session error: {e}")
            raise
        finally:
            self.cleanup_resources()


class QuietDQNPlayer(LocalDQNPlayer):
    """Version with minimal console output for batch processing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def play_single_episode(self, episode_number):
        """Play episode with minimal output."""
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        episode_done = False
        
        while not episode_done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            
            total_reward += reward[0]
            steps += 1
            
            if self.save_video:
                frame = self.env.render()
                if frame is not None:
                    if self.video_writer is None:
                        self.setup_video_writer(frame)
                    self.save_frame_to_video(frame)
            
            time.sleep(self.delay_between_actions)
            
            if done[0]:
                episode_done = True
        
        return total_reward, steps

    def play_all_episodes(self):
        """Play all episodes quietly."""
        self.episode_rewards = []
        self.episode_lengths = []
        
        try:
            for episode in range(1, self.num_episodes + 1):
                episode_reward, episode_length = self.play_single_episode(episode)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error: {e}")


def play_dqn_agent(env_name="ALE/Boxing-v5", model_path="dqn_model.zip", 
                   episodes=5, save_video=False, quiet_mode=False):
    """
    Convenience function to play a trained DQN agent.
    
    Args:
        env_name: Atari environment name
        model_path: Path to trained model file
        episodes: Number of episodes to play
        save_video: Whether to record video
        quiet_mode: Minimal output mode
    
    Returns:
        Performance statistics
    """
    if quiet_mode:
        player = QuietDQNPlayer(
            env_name=env_name,
            model_path=model_path,
            num_episodes=episodes,
            save_video=save_video
        )
    else:
        player = LocalDQNPlayer(
            env_name=env_name,
            model_path=model_path,
            num_episodes=episodes,
            save_video=save_video
        )
    
    return player.run_gameplay_session()


def main():
    """Main function for standalone execution."""
    # Configuration
    ENV_NAME = "ALE/Boxing-v5"
    MODEL_PATH = "dqn_model.zip"
    NUM_EPISODES = 3
    SAVE_VIDEO = False
    QUIET_MODE = False
    
    print("DQN Agent Local Player")
    print("Make sure you have a display available for the game window")
    
    # Run the gameplay session
    stats = play_dqn_agent(
        env_name=ENV_NAME,
        model_path=MODEL_PATH,
        episodes=NUM_EPISODES,
        save_video=SAVE_VIDEO,
        quiet_mode=QUIET_MODE
    )
    
    return stats


if __name__ == "__main__":
    results = main()
