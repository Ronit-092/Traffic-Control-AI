"""train.py — Train PPO. Run: python train.py"""

import os, warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium_adapter import TrafficGymnasiumEnv

os.makedirs("models/best", exist_ok=True)
os.makedirs("logs",        exist_ok=True)


def train():
    print("=" * 50)
    print("  Traffic-AI — PPO Training")
    print("=" * 50)

    env   = Monitor(TrafficGymnasiumEnv(), "logs/")
    model = PPO(
        "MlpPolicy", env,
        n_steps=2048, batch_size=64, learning_rate=2.5e-4,
        clip_range=0.15, ent_coef=0.01, gae_lambda=0.95,
        gamma=0.99, n_epochs=10,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1, seed=42,
    )

    print("\n  Training 50,000 steps (~3-5 min)...\n")
    model.learn(total_timesteps=50_000, progress_bar=True)

    model.save("models/traffic_model")
    model.save("models/best/best_model")
    print("\n✅ models/traffic_model.zip")
    print("✅ models/best/best_model.zip")
    env.close()


if __name__ == "__main__":
    train()
