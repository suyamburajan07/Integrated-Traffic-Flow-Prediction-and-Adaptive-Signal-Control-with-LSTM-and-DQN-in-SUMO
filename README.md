# Adaptive Traffic Signal Control using LSTM and Deep Reinforcement Learning

This repository contains the implementation of an AI-driven smart traffic signal control system. It integrates **LSTM-based traffic flow forecasting** with a **Deep Q-Network (DQN)** reinforcement learning agent for real-time signal optimization in a simulated traffic environment using **SUMO**.

---

## Project Overview

Urban traffic congestion is a major challenge due to growing population and infrastructure limits. This project proposes a modular framework that:

- Predicts short-term traffic flow using **LSTM**
- Uses predictions to train a **DQN agent** to optimize signal timing
- Simulates traffic behavior and control using **SUMO (Simulation of Urban Mobility)**

---

## System Architecture

METR-LA Dataset ──► LSTM Model ──► Forecasted Queues
      │
RL Agent (DQN)
      │
SUMO Traffic Simulation
      │
Queue Evaluation Logs

---

## 🗂️ Project Structure

📦 traffic-signal-rl
├── data/                       # Preprocessed dataset / forecast.csv
├── model/                      # Saved models 
├── sumo_sim/                   # SUMO network and route files
├── scripts/
│ ├── train_lstm.py             # LSTM model training
│ ├── train_dqn.py              # DQN agent training
│ ├── eval_dqn.py               # RL agent evaluation
│ ├── run_fixed.py              # Fixed-timing baseline simulation
│ └── sumo_env.py               # Custom Gym + TraCI environment
├── rl_queue.csv                # Logged queue lengths from RL
├── baseline_queue.csv          # Logged queue lengths from fixed-time controller
├── README.md
└── Requirements.txt

---

## 📊 Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** — for LSTM prediction
- **Stable-Baselines3** — for DQN training
- **OpenAI Gym** — environment wrapping
- **SUMO (Simulation of Urban Mobility)** — microscopic traffic simulation
- **TraCI** — real-time SUMO control interface

---