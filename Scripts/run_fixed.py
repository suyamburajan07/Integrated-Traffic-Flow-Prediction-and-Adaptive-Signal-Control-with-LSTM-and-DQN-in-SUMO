import traci
import csv

sumo_binary = "sumo"
sumo_config = "sim.sumocfg"
lane_ids = ["2_0", "4_0", "3_0", "1_0"]

traci.start([sumo_binary, "-c", sumo_config, "--start"])

with open("baseline_queue.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "lane_1", "lane_2", "lane_3", "lane_4"])

    for step in range(200):
        traci.simulationStep()
        queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids]
        writer.writerow([step] + queues)

traci.close()
print("✅ Fixed-time baseline simulation complete. Logged to baseline_queue.csv")
