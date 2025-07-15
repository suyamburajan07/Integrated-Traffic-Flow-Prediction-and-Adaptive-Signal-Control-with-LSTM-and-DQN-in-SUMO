import gym
from gym import spaces
import numpy as np
import traci
import pandas as pd
import csv

class SumoTrafficEnv(gym.Env):
    def __init__(self):
        super(SumoTrafficEnv, self).__init__()
        self.lanes = ["2_0", "4_0", "3_0", "1_0"]
        self.action_space = spaces.Discrete(4)  # 4 signal phases
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = 200

        # Load forecasted traffic
        self.forecast = pd.read_csv("forecast.csv").values
        self.forecast_step = 0

        self.queue_log = open("rl_queue.csv", "w", newline="")
        self.writer = csv.writer(self.queue_log)
        self.writer.writerow(["step", "lane_1", "lane_2", "lane_3", "lane_4"])

    def reset(self):
        traci.start(["sumo", "-c", "sim.sumocfg", "--start"])
        self.step_count = 0
        self.forecast_step = 0
        return self._get_state()

    # def _get_state(self):
    #     # Option 1: Live queue from SUMO
    #     queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]
    #
    #     # Option 2: Forecasted queue (LSTM)
    #     if self.forecast_step < len(self.forecast):
    #         queues = self.forecast[self.forecast_step]
    #         self.forecast_step += 1
    #     return np.array(queues, dtype=np.float32)

    def _get_state(self):
        # Option 1: Live queue from SUMO
        queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]

        # Option 2: Forecasted queue (LSTM)
        if self.forecast_step < len(self.forecast):
            queues = self.forecast[self.forecast_step]
            self.forecast_step += 1
        return np.array(queues, dtype=np.float32)

    def step(self, action):
        if self.step_count >= self.max_steps:
            traci.close()
            return np.zeros(4), 0, True, {}

        self._apply_action(action)
        traci.simulationStep()
        self.step_count += 1
        state = self._get_state()
        self.writer.writerow([self.step_count] + list(state))

        reward = -sum(state)
        done = self.step_count >= self.max_steps

        if done:
            traci.close()

        return state, reward, done, {}

    def _apply_action(self, action):
        tls_id = traci.trafficlight.getIDList()[0]
        phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases
        traci.trafficlight.setPhase(tls_id, action)

    def close(self):
        self.queue_log.close()
        traci.close()

