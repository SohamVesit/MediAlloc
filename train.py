import numpy as np
import json
from environment import HospitalEnv
from agent import QLearningAgent

env   = HospitalEnv()
agent = QLearningAgent()

NUM_EPISODES      = 2000
rewards_log       = []
treated_log       = []

print("🏥 Training Q-Learning Agent...")

for ep in range(NUM_EPISODES):
    state       = env.reset()
    total_reward = 0
    done         = False

    while not done:
        action                  = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state        = next_state
        total_reward += reward

    rewards_log.append(total_reward)
    treated_log.append(env.patients_treated)

    if ep % 200 == 0:
        avg = np.mean(rewards_log[-200:])
        print(f"  Episode {ep:4d} | ε={agent.epsilon:.3f} | Avg Reward: {avg:.1f}")

agent.save("model.pkl")

# Save training metrics for frontend charts
with open("training_stats.json", "w") as f:
    json.dump({
        "rewards":  rewards_log,
        "treated":  treated_log,
        "episodes": NUM_EPISODES
    }, f)

print("✅ Training complete! model.pkl + training_stats.json saved.")