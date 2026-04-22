const API = "http://localhost:5000/api";

// ── Chart Setup ───────────────────────────────────────────────────
const chartCfg = (label, color) => ({
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label,
      data: [],
      borderColor: color,
      backgroundColor: color + "22",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.4,
      fill: true,
    }]
  },
  options: {
    animation: false,
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { ticks: { color: "#64748b", font: { family: "Space Mono", size: 10 } },
           grid:  { color: "#1e2d45" } }
    }
  }
});

const rewardChart  = new Chart(document.getElementById("chart-reward"),
                                chartCfg("Reward", "#00d4ff"));
const treatedChart = new Chart(document.getElementById("chart-treated"),
                                chartCfg("Treated", "#22c55e"));
const trainingChart = new Chart(document.getElementById("chart-training"), {
  type: "line",
  data: { labels: [], datasets: [{
    label: "Episode Reward",
    data: [], borderColor: "#ff6b35",
    backgroundColor: "#ff6b3522", borderWidth: 1.5,
    pointRadius: 0, tension: 0.3, fill: true,
  }]},
  options: {
    animation: false, responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { ticks: { color: "#64748b", font: { family: "Space Mono", size: 10 } },
           grid:  { color: "#1e2d45" } }
    }
  }
});

// ── Helpers ───────────────────────────────────────────────────────
function pushChart(chart, value, maxPoints = 60) {
  chart.data.labels.push("");
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > maxPoints) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update();
}

function updateResourceBar(id, value, max) {
  const pct  = value / max;
  const bar  = document.getElementById("bar-" + id);
  const val  = document.getElementById("val-" + id);
  bar.style.width = (pct * 100) + "%";
  bar.className = "res-bar " + (pct < 0.3 ? "low" : pct < 0.6 ? "mid" : "");
  val.textContent = value;
}

function addLog(text, type = "info") {
  const log  = document.getElementById("event-log");
  const div  = document.createElement("div");
  div.className = "log-entry log-" + type;
  const time = new Date().toLocaleTimeString("en", { hour12: false });
  div.textContent = `[${time}] ${text}`;
  log.prepend(div);
  if (log.children.length > 80) log.lastChild.remove();
}

function setStatus(mode) {
  const pill = document.getElementById("status-pill");
  pill.className = "status-pill " + mode;
  pill.textContent = mode === "running" ? "● RUNNING"
                   : mode === "done"    ? "● DONE"
                   :                      "● IDLE";
}

function updateState(s) {
  updateResourceBar("icu", s.icu_beds,   s.max_icu);
  updateResourceBar("gen", s.gen_beds,   s.max_gen_beds);
  updateResourceBar("doc", s.doctors,    s.max_doctors);
  updateResourceBar("amb", s.ambulances, s.max_ambulances);

  document.getElementById("p-critical").textContent = s.critical_patients;
  document.getElementById("p-moderate").textContent = s.moderate_patients;
  document.getElementById("p-mild").textContent     = s.mild_patients;
  document.getElementById("stat-step").textContent  = s.timestep;
  document.getElementById("stat-treated").textContent = s.patients_treated;
  document.getElementById("stat-died").textContent    = s.patients_died;
}

function setActionBanner(event, reward) {
  const banner = document.getElementById("action-banner");
  banner.textContent = event;
  banner.className = "action-banner " +
    (reward >= 15 ? "positive" : reward < 0 ? "negative" : "neutral");
}

// ── API Calls ─────────────────────────────────────────────────────
let autoInterval = null;
let autoRunning  = false;

async function startEpisode() {
  const res  = await fetch(API + "/start", { method: "POST" });
  const data = await res.json();
  updateState(data.state);

  rewardChart.data.labels  = [];
  rewardChart.data.datasets[0].data  = [];
  treatedChart.data.labels = [];
  treatedChart.data.datasets[0].data = [];
  rewardChart.update();
  treatedChart.update();

  document.getElementById("stat-reward").textContent = "0";
  document.getElementById("btn-step").disabled  = false;
  document.getElementById("btn-auto").disabled  = false;
  setStatus("running");
  addLog("Episode started — agent ready", "info");
}

async function stepOnce() {
  const res  = await fetch(API + "/step", { method: "POST" });
  if (!res.ok) { addLog("Error — start an episode first", "danger"); return; }
  const data = await res.json();

  updateState(data.state);
  setActionBanner(data.event, data.reward);
  document.getElementById("stat-reward").textContent = data.total_reward;
  document.getElementById("epsilon-val").textContent =
    (data.epsilon ?? 0.00).toFixed(3);

  pushChart(rewardChart,  data.reward);
  pushChart(treatedChart, data.state.patients_treated);

  const logType = data.reward >= 15 ? "success"
                : data.reward < 0   ? "danger"
                : "warn";
  addLog(`${data.action_name} → reward ${data.reward > 0 ? "+" : ""}${data.reward}`, logType);

  if (data.done) {
    setStatus("done");
    addLog(`Episode complete — total reward: ${data.total_reward}`, "info");
    document.getElementById("btn-step").disabled = true;
    if (autoRunning) toggleAuto();
  }
}

function toggleAuto() {
  autoRunning = !autoRunning;
  const btn   = document.getElementById("btn-auto");
  if (autoRunning) {
    btn.textContent = "⏸ Pause";
    autoInterval    = setInterval(stepOnce, 300);
  } else {
    btn.textContent = "⚡ Auto Run";
    clearInterval(autoInterval);
  }
}

async function resetSim() {
  clearInterval(autoInterval);
  autoRunning = false;
  document.getElementById("btn-auto").textContent = "⚡ Auto Run";
  await fetch(API + "/reset", { method: "POST" });
  document.getElementById("btn-step").disabled = true;
  document.getElementById("btn-auto").disabled = true;
  setStatus("idle");
  setActionBanner("🤖 Agent is ready — press Start Episode", 0);
  addLog("Simulation reset", "info");
}

async function loadTrainingStats() {
  const res  = await fetch(API + "/training-stats");
  if (!res.ok) { addLog("No training stats found — run train.py first", "danger"); return; }
  const data = await res.json();

  // Downsample to 200 points for performance
  const rewards = data.rewards;
  const step    = Math.max(1, Math.floor(rewards.length / 200));
  const sampled = rewards.filter((_, i) => i % step === 0);

  trainingChart.data.labels = sampled.map((_, i) => i * step);
  trainingChart.data.datasets[0].data = sampled;
  trainingChart.update();
  addLog(`Training stats loaded — ${data.episodes} episodes`, "success");
}