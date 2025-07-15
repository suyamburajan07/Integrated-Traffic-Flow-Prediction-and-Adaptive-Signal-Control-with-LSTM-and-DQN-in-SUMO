from stable_baselines3 import DQN
from sumo_env import SumoTrafficEnv

env = SumoTrafficEnv()
model = DQN.load("models/dqn_traffic_model")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

env.close()
print("✅ DQN agent evaluation complete. Logged to rl_queue.csv")
