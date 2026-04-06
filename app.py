@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>

<title>AI Traffic Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body {
    margin:0;
    font-family: 'Segoe UI', sans-serif;
    background: radial-gradient(circle at top, #0f172a, #020617);
    color:white;
    text-align:center;
}

/* HEADER */
h1 {
    font-size: 32px;
    margin-top: 20px;
    background: linear-gradient(90deg, #22c55e, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CONTROLS */
.controls {
    margin: 15px;
}

button, select {
    padding:10px 15px;
    border-radius:10px;
    border:none;
    margin:5px;
    cursor:pointer;
    font-weight:bold;
    transition:0.3s;
}

button:hover {
    transform:scale(1.1);
}

.reset { background:#22c55e; }
.auto { background:#3b82f6; }
.stop { background:#ef4444; }

/* DASHBOARD GRID */
.dashboard {
    display:flex;
    justify-content:center;
    gap:20px;
    margin-top:20px;
}

/* ROAD */
.road {
    position:relative;
    width:300px;
    height:300px;
}

/* LANES */
.lane {
    position:absolute;
    display:flex;
    gap:5px;
}

.north { top:0; left:140px; flex-direction:column-reverse; }
.south { bottom:0; left:140px; flex-direction:column; }
.east  { right:0; top:140px; }
.west  { left:0; top:140px; flex-direction:row-reverse; }

/* CAR */
.car {
    width:10px;
    height:10px;
    border-radius:3px;
    background:#22c55e;
    box-shadow:0 0 8px #22c55e;
}

/* SIGNAL */
.signal {
    position:absolute;
    top:130px;
    left:130px;
    width:40px;
    height:40px;
    border-radius:10px;
    transition:0.3s;
}

.green {
    background:#22c55e;
    box-shadow:0 0 20px #22c55e;
}

.red {
    background:#ef4444;
}

/* STATS PANEL */
.panel {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding:20px;
    border-radius:15px;
    width:250px;
    text-align:left;
}

.panel p {
    margin:8px 0;
}

/* CHART */
.chart {
    width:700px;
    margin:30px auto;
}

</style>
</head>

<body>

<h1>🚦 AI Traffic Dashboard</h1>

<div class="controls">
<select id="difficulty">
<option value="easy">Easy</option>
<option value="medium">Medium</option>
<option value="hard">Hard</option>
</select>

<button class="reset" onclick="reset()">Reset</button>
<button class="auto" onclick="startAuto()">AI Auto</button>
<button class="stop" onclick="stopAuto()">Stop</button>

<br>
Speed:
<input type="range" id="speed" min="100" max="1000" value="300">
</div>

<div class="dashboard">

<div class="road">
<div id="north" class="lane north"></div>
<div id="south" class="lane south"></div>
<div id="east" class="lane east"></div>
<div id="west" class="lane west"></div>
<div id="signal" class="signal"></div>
</div>

<div class="panel">
<p>🚗 Queue: <span id="queue">0</span></p>
<p>⏱ Steps: <span id="steps">0</span></p>
<p>💰 Reward: <span id="total">0</span></p>
<p>📊 Avg Wait: <span id="avg">0</span></p>
<p>🚀 Throughput: <span id="throughput">0</span></p>
<p>⚡ Latency: <span id="latency">0 ms</span></p>
</div>

</div>

<div>
<button onclick="manual(0)">↑</button>
<button onclick="manual(1)">↓</button>
<button onclick="manual(2)">→</button>
<button onclick="manual(3)">←</button>
</div>

<canvas id="chart" class="chart"></canvas>

<script>

let steps = 0;
let total = 0;
let running = false;
let loop;

let chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderColor:"#22c55e"}]}
});

function drawCars(id, count) {
    let lane = document.getElementById(id);
    lane.innerHTML = "";
    for (let i = 0; i < count; i++) {
        let car = document.createElement("div");
        car.className = "car";
        lane.appendChild(car);
    }
}

function update(obs, reward, latency) {

    drawCars("north", obs.north_queue);
    drawCars("south", obs.south_queue);
    drawCars("east", obs.east_queue);
    drawCars("west", obs.west_queue);

    document.getElementById("signal").className =
        "signal green";

    let q = obs.north_queue + obs.south_queue + obs.east_queue + obs.west_queue;

    document.getElementById("queue").innerText = q;
    document.getElementById("steps").innerText = steps;
    document.getElementById("total").innerText = total.toFixed(2);
    document.getElementById("avg").innerText =
        (obs.total_waiting / (steps + 1)).toFixed(2);
    document.getElementById("throughput").innerText = obs.throughput;
    document.getElementById("latency").innerText = latency + " ms";

    chart.data.labels.push(steps);
    chart.data.datasets[0].data.push(reward);
    chart.update();
}

async function reset() {
    let level = document.getElementById("difficulty").value;

    let start = performance.now();
    let res = await fetch("/reset?level=" + level, {method:"POST"});
    let latency = Math.round(performance.now() - start);

    let data = await res.json();

    steps = 0;
    total = 0;

    chart.data.labels = [];
    chart.data.datasets[0].data = [];

    update(data, 0, latency);
}

async function manual(a) {

    let start = performance.now();

    let res = await fetch("/step", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({signal:a})
    });

    let latency = Math.round(performance.now() - start);

    let data = await res.json();

    steps++;
    total += data.reward;

    update(data.observation, data.reward, latency);
}

async function aiStep() {

    let start = performance.now();

    let res = await fetch("/ai-step");
    let latency = Math.round(performance.now() - start);

    let data = await res.json();

    steps++;
    total += data.reward;

    update(data.observation, data.reward, latency);
}

function startAuto() {
    if (running) return;
    running = true;
    let speed = document.getElementById("speed").value;
    loop = setInterval(aiStep, speed);
}

function stopAuto() {
    running = false;
    clearInterval(loop);
}

</script>

</body>
</html>
"""