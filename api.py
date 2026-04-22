from flask import Flask, jsonify, request
from flask_cors import CORS
import json, os
from environment import HospitalEnv
from agent import QLearningAgent

app   = Flask(__name__)
CORS(app)

env   = HospitalEnv()
agent = QLearningAgent()
agent.load("model.pkl")

ACTION_NAMES = [
    "Assign ICU bed to critical patient",
    "Assign general bed to moderate patient",
    "Dispatch ambulance",
    "Assign doctor to mild patient",
    "Queue patient"
]

# ── Global session state ──────────────────────────────────────────
current_state    = None
episode_rewards  = []
current_reward   = 0
is_running       = False

@app.route("/api/start", methods=["POST"])
def start_episode():
    global current_state, episode_rewards, current_reward, is_running
    current_state   = env.reset()
    episode_rewards = []
    current_reward  = 0
    is_running      = True
    return jsonify({"status": "started", "state": env.get_full_state()})

@app.route("/api/step", methods=["POST"])
def step():
    global current_state, current_reward, is_running
    if not is_running:
        return jsonify({"error": "No episode running. Call /api/start first."}), 400

    action                           = agent.choose_action(current_state)
    next_state, reward, done, event  = env.step(action)

    current_state   = next_state
    current_reward += reward
    episode_rewards.append(reward)

    if done:
        is_running = False

    return jsonify({
        "action":      action,
        "action_name": ACTION_NAMES[action],
        "reward":      reward,
        "total_reward": current_reward,
        "done":        done,
        "event":       event,
        "state":       env.get_full_state(),
        "history":     env.history[-1],
    })

@app.route("/api/training-stats", methods=["GET"])
def training_stats():
    if os.path.exists("training_stats.json"):
        with open("training_stats.json") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Train the model first!"}), 404

@app.route("/api/reset", methods=["POST"])
def reset():
    global is_running
    is_running = False
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)