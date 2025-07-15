from sumo_env import SumoTrafficEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os

# Create environment
env = SumoTrafficEnv()

# Initialize DQN model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0005, buffer_size=10000)


class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=1):
        super(EpisodeLogger, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        done_array = self.locals["dones"]
        if any(done_array):
            self.episode_count += 1
            print(f"🔁 Episode {self.episode_count} complete")
        return True

# Train for 10 episodes (you can increase this)
model.learn(total_timesteps=2000, callback=EpisodeLogger())

# Save the model
#model.save("dqn_traffic_model")


os.makedirs("models", exist_ok=True)
print("✅ Training finished.")
model.save("models/dqn_traffic_model")
#print("✅ Model trained and saved.")