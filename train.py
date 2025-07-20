import os
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed
import shutil
import ale_py


class DQNTrainer:
    """DQN training pipeline with hyperparameter tuning and policy comparison."""
    
    def __init__(self, env_name="ALE/Boxing-v5", total_timesteps=500000):
        self.env_name = env_name
        self.total_timesteps = total_timesteps
        self.default_hyperparams = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.1,
            'buffer_size': 50000,
            'learning_starts': 10000,
            'target_update_interval': 2500,
            'train_freq': 4,
            'gradient_steps': 1
        }
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for results, models, and logs."""
        directories = ["results", "models", "logs"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def create_environments(self, seed=0, eval_seed=42):
        """Create training and evaluation environments with frame stacking."""
        train_env = make_atari_env(self.env_name, n_envs=1, seed=seed)
        train_env = VecFrameStack(train_env, n_stack=4)
        
        eval_env = make_atari_env(self.env_name, n_envs=1, seed=eval_seed)
        eval_env = VecFrameStack(eval_env, n_stack=4)
        
        return train_env, eval_env

    def create_model(self, policy, hyperparams, train_env, tensorboard_log, verbose=1):
        """Create a DQN model with specified parameters."""
        return DQN(
            policy=policy,
            env=train_env,
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            batch_size=hyperparams['batch_size'],
            exploration_initial_eps=hyperparams['epsilon_start'],
            exploration_final_eps=hyperparams['epsilon_end'],
            exploration_fraction=hyperparams['epsilon_decay'],
            buffer_size=hyperparams.get('buffer_size', 50000),
            learning_starts=hyperparams.get('learning_starts', 10000),
            target_update_interval=hyperparams.get('target_update_interval', 2500),
            train_freq=hyperparams.get('train_freq', 4),
            gradient_steps=hyperparams.get('gradient_steps', 1),
            verbose=verbose,
            tensorboard_log=tensorboard_log
        )

    def create_callbacks(self, eval_env, save_path, log_dir, eval_freq=2500, 
                        max_no_improvement=8, min_evals=7, n_eval_episodes=5):
        """Create evaluation and early stopping callbacks."""
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=max_no_improvement,
            min_evals=min_evals,
            verbose=1 if eval_freq >= 5000 else 0
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=n_eval_episodes,
            verbose=1 if eval_freq >= 5000 else 0,
            callback_after_eval=early_stop_callback
        )

        return eval_callback

    def train_single_model(self, policy, hyperparams, timesteps, save_path, 
                          log_dir, seed=0, progress_bar=True):
        """Train a single DQN model and return results."""
        try:
            # Setup directories
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            # Create environments
            train_env, eval_env = self.create_environments(seed=seed, eval_seed=seed+42)

            # Create model
            model = self.create_model(policy, hyperparams, train_env, log_dir)

            # Setup callbacks
            eval_freq = 2500 if timesteps >= self.total_timesteps // 2 else 2500
            eval_callback = self.create_callbacks(eval_env, save_path, log_dir, eval_freq)

            # Model path
            model_name = f"dqn_{self.env_name.replace('/', '_')}_{policy}"
            model_path = os.path.join(save_path, f"{model_name}.zip")

            print(f"Training DQN with {policy} policy")
            print(f"Hyperparameters: {hyperparams}")

            # Train the model
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                tb_log_name=f"DQN_{policy}",
                progress_bar=progress_bar
            )

            # Save model
            model.save(model_path)
            
            result = {
                'model_path': model_path,
                'best_reward': float(eval_callback.best_mean_reward),
                'hyperparams': hyperparams
            }

            # Cleanup
            train_env.close()
            eval_env.close()

            return result

        except Exception as e:
            print(f"Training failed: {e}")
            return {'error': str(e)}

    def compare_policies(self):
        """Compare MLP and CNN policies and return the best one."""
        print("COMPARING POLICIES (MLP vs CNN)")
        
        policies = ["MlpPolicy", "CnnPolicy"]
        policy_results = {}

        for policy in policies:
            print(f"\nTraining with {policy}")
            
            save_path = f"models/{policy.lower()}"
            log_dir = f"logs/{policy.lower()}"
            
            result = self.train_single_model(
                policy=policy,
                hyperparams=self.default_hyperparams,
                timesteps=self.total_timesteps // 2,
                save_path=save_path,
                log_dir=log_dir,
                seed=0
            )
            
            policy_results[policy] = result

        # Save results
        with open("results/policy_comparison.json", "w") as f:
            json.dump(policy_results, f, indent=2)

        # Determine best policy
        best_policy = "CnnPolicy"  # Default for Atari
        if all(policy in policy_results and 'best_reward' in policy_results[policy] 
               for policy in policies):
            mlp_reward = policy_results["MlpPolicy"]["best_reward"]
            cnn_reward = policy_results["CnnPolicy"]["best_reward"]
            if mlp_reward > cnn_reward:
                best_policy = "MlpPolicy"

        print(f"\nBest policy: {best_policy}")
        return best_policy, policy_results

    def _tune_single_hyperparameter(self, trial_idx, params_dict, policy):
        """Train a single hyperparameter configuration (for parallel execution)."""
        print(f"\nTrial {trial_idx + 1}: {params_dict}")

        save_path = f"models/tuning_trial_{trial_idx}"
        log_dir = f"logs/tuning_trial_{trial_idx}"

        result = self.train_single_model(
            policy=policy,
            hyperparams=params_dict,
            timesteps=self.total_timesteps // 4,
            save_path=save_path,
            log_dir=log_dir,
            seed=trial_idx,
            progress_bar=False
        )

        if 'best_reward' in result:
            result.update({
                'trial': trial_idx,
                'params': params_dict
            })
            print(f"Trial {trial_idx + 1} completed. Best reward: {result['best_reward']:.2f}")
        
        return result

    def tune_hyperparameters(self, policy):
        """Perform grid search hyperparameter tuning."""
        print(f"HYPERPARAMETER TUNING WITH {policy}")

        # Define parameter grid
        param_grid = {
            'learning_rate': [1e-3, 3e-4],
            'gamma': [0.99, 0.995],
            'batch_size': [16, 32],
            'epsilon_start': [1.0],
            'epsilon_end': [0.05, 0.1],
            'epsilon_decay': [0.1, 0.2]
        }

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        print(f"Testing {len(combinations)} combinations")
        print(f"Each trial: {self.total_timesteps // 4:,} timesteps")

        # Create parameter dictionaries for each combination
        param_combinations = []
        for combination in combinations:
            params = dict(zip(keys, combination))
            # Add default values for parameters not being tuned
            for key, value in self.default_hyperparams.items():
                if key not in params:
                    params[key] = value
            param_combinations.append(params)

        # Parallel hyperparameter tuning
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(self._tune_single_hyperparameter)(i, params, policy)
            for i, params in enumerate(param_combinations)
        )

        # Process results
        successful_results = [r for r in results if 'best_reward' in r and 'error' not in r]
        
        best_params = self.default_hyperparams
        best_reward = float('-inf')
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['best_reward'])
            best_reward = best_result['best_reward']
            best_params = best_result['params']

        # Save tuning results
        tuning_data = {
            'best_params': best_params,
            'best_reward': best_reward,
            'all_results': successful_results
        }

        with open("results/hyperparameter_tuning.json", "w") as f:
            json.dump(tuning_data, f, indent=2)

        # Create and save results table
        if successful_results:
            self._save_tuning_table(successful_results)

        return best_params, tuning_data

    def _save_tuning_table(self, results):
        """Create and save hyperparameter tuning results table."""
        # Sort results by reward
        sorted_results = sorted(results, key=lambda x: x['best_reward'], reverse=True)

        # Create table
        table_lines = [
            "Hyperparameter Tuning Results",
            "=" * 80,
            f"{'Trial':<6} {'LR':<8} {'Gamma':<6} {'Batch':<6} {'EpsE':<6} {'EpsD':<6} {'Reward':<8}",
            "-" * 80
        ]

        for result in sorted_results[:10]:  # Top 10 results
            params = result['params']
            line = (f"{result['trial']:<6} {params['learning_rate']:<8} "
                   f"{params['gamma']:<6} {params['batch_size']:<6} "
                   f"{params['epsilon_end']:<6} {params['epsilon_decay']:<6} "
                   f"{result['best_reward']:<8.2f}")
            table_lines.append(line)

        table_str = "\n".join(table_lines)

        # Save table
        with open("results/hyperparameter_table.txt", "w") as f:
            f.write(table_str)

        print("\nTop 10 Hyperparameter Configurations:")
        print(table_str)

    def final_training(self, policy, hyperparams):
        """Perform final training with best configuration."""
        print("FINAL TRAINING WITH BEST CONFIGURATION")

        save_path = "models/final"
        log_dir = "logs/final"

        result = self.train_single_model(
            policy=policy,
            hyperparams=hyperparams,
            timesteps=self.total_timesteps,
            save_path=save_path,
            log_dir=log_dir,
            seed=0
        )

        if 'model_path' in result:
            # Copy to standard location
            shutil.copy(result['model_path'], "dqn_model.zip")
            
            # Save final results
            final_results = {
                'env_name': self.env_name,
                'policy_type': policy,
                'hyperparameters': hyperparams,
                'total_timesteps': self.total_timesteps,
                'best_reward': result['best_reward'],
                'model_path': result['model_path']
            }

            with open("results/final_results.json", "w") as f:
                json.dump(final_results, f, indent=2)

            print(f"Final model saved as: dqn_model.zip")
            print(f"Best reward: {result['best_reward']:.2f}")

        return result

    def create_evaluation_plot(self):
        """Create and save evaluation performance plot."""
        try:
            eval_log_path = "logs/final/evaluations.npz"
            if not os.path.exists(eval_log_path):
                print("No evaluation data found for plotting")
                return

            # Load evaluation data
            data = np.load(eval_log_path)
            timesteps = data["timesteps"]
            results = data["results"]

            # Compute statistics
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)

            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, mean_rewards, label="Mean Evaluation Reward")
            plt.fill_between(timesteps, mean_rewards - std_rewards, 
                           mean_rewards + std_rewards, alpha=0.2, label="±1 Std Dev")
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title(f"DQN Evaluation Performance on {self.env_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("results/evaluation_curve.png")
            plt.show()

        except Exception as e:
            print(f"Could not create evaluation plot: {e}")

    def run_complete_pipeline(self, compare_policies=True, tune_hyperparameters=True):
        """Run the complete DQN training pipeline."""
        print(f"Training DQN on {self.env_name}")
        print(f"Total timesteps: {self.total_timesteps}")

        # Step 1: Policy comparison
        if compare_policies:
            best_policy, policy_results = self.compare_policies()
        else:
            best_policy = "CnnPolicy"  # Default for Atari

        # Step 2: Hyperparameter tuning
        if tune_hyperparameters:
            best_hyperparams, tuning_results = self.tune_hyperparameters(best_policy)
        else:
            best_hyperparams = self.default_hyperparams

        # Step 3: Final training
        final_result = self.final_training(best_policy, best_hyperparams)

        # Step 4: Create visualization
        self.create_evaluation_plot()

        # Summary
        print(f"\nTraining Complete!")
        print(f"Best policy: {best_policy}")
        print(f"Best hyperparameters: {best_hyperparams}")
        if 'best_reward' in final_result:
            print(f"Final best reward: {final_result['best_reward']:.2f}")
        print(f"All results saved in: results/")

        return {
            'best_policy': best_policy,
            'best_hyperparams': best_hyperparams,
            'final_result': final_result
        }


def main():
    """Main execution function."""
    # Configuration
    ENV_NAME = "ALE/Boxing-v5"
    TOTAL_TIMESTEPS = 500000
    RUN_HYPERPARAMETER_TUNING = True
    COMPARE_POLICIES = True

    # Create trainer and run pipeline
    trainer = DQNTrainer(env_name=ENV_NAME, total_timesteps=TOTAL_TIMESTEPS)
    results = trainer.run_complete_pipeline(
        compare_policies=COMPARE_POLICIES,
        tune_hyperparameters=RUN_HYPERPARAMETER_TUNING
    )

    return results


if __name__ == "__main__":
    results = main()
