const $ = (s) => document.querySelector(s);
const FAKE = "#ff5470", REAL = "#2ed39a", ACC = "#6c8cff";
Chart.defaults.color = "#8b97ab"; Chart.defaults.borderColor = "#222c3d";
let charts = {}, file = null;

// ---------- tabs
document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tab,.page").forEach(e => e.classList.remove("active"));
  b.classList.add("active"); $("#" + b.dataset.tab).classList.add("active");
});

// ---------- upload
const drop = $("#drop");
function setFile(f) {
  if (!f || !f.type.startsWith("video")) return;
  file = f; const v = $("#preview");
  v.src = URL.createObjectURL(f); v.hidden = false; $("#dropText").hidden = true; $("#go").disabled = false;
}
$("#file").onchange = e => setFile(e.target.files[0]);
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

// ---------- helpers
const pretty = k => k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3);
const MODEL_NAMES = { logreg: "Logistic Reg.", svm: "SVM (RBF)", lightgbm: "LightGBM", xgboost: "XGBoost", fusion_net: "Fusion Transformer", cgtf: "CGTF (tri-modal)", fusion_cgtf: "Fusion Transformer (CGTF)", block_stack: "Block-wise late fusion" };
function table(el, obj) {
  $(el).innerHTML = Object.entries(obj).map(([k, v]) => `<tr><td>${pretty(k)}</td><td>${fmt(v)}</td></tr>`).join("");
}
function mk(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); }

function gauge(p, thr) {
  const c = $("#gauge").getContext("2d"), col = p >= thr ? FAKE : REAL;
  c.clearRect(0, 0, 260, 260); c.lineWidth = 18; c.lineCap = "round";
  c.strokeStyle = "#222c3d"; c.beginPath(); c.arc(130, 130, 105, 0.75 * Math.PI, 2.25 * Math.PI); c.stroke();
  c.strokeStyle = col; c.beginPath(); c.arc(130, 130, 105, 0.75 * Math.PI, (0.75 + 1.5 * p) * Math.PI); c.stroke();
  const a = (0.75 + 1.5 * thr) * Math.PI; c.strokeStyle = "#fff"; c.lineWidth = 3;
  c.beginPath(); c.moveTo(130 + 92 * Math.cos(a), 130 + 92 * Math.sin(a)); c.lineTo(130 + 118 * Math.cos(a), 130 + 118 * Math.sin(a)); c.stroke();
}

// ---------- analyze
$("#go").onclick = async () => {
  const fd = new FormData();
  fd.append("video", file); fd.append("caption", $("#caption").value); fd.append("verified", $("#verified").checked ? 1 : 0); fd.append("bio", $("#bio").value);
  $("#go").disabled = true; $("#status").innerHTML = '<span class="spinner"></span>' + (MI && MI.version === 3 ? "Extracting frames, running CLIP + MiniLM and 6 models…" : "Extracting frames, transcribing speech, running CLIP + CLAP and 5 models…");
  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const d = await r.json(); if (!r.ok) throw new Error(d.detail || "Failed");
    render(d); $("#status").textContent = "Done.";
  } catch (e) { $("#status").textContent = "Error: " + e.message; }
  $("#go").disabled = false;
};

function render(d) {
  $("#emptyState").hidden = true; $("#verdictBody").hidden = false; $("#details").hidden = false;
  const fake = d.verdict === "FAKE";
  $("#verdict").textContent = d.verdict; $("#verdict").className = "verdict " + (fake ? "fake" : "real");
  $("#prob").textContent = `Fake probability ${(d.fake_probability * 100).toFixed(1)}%`;
  gauge(d.fake_probability, d.threshold);
  const votes = Object.values(d.model_probs).filter(p => p >= d.threshold).length;
  const cons = d.consistency.clip_sim_mean;
  const modeTxt = d.mode === "video_only" ? ' <b>Video-only mode</b> (no caption used).' : "";
  const consTxt = cons !== undefined ? ` Caption-video consistency is <b>${cons > 0.27 ? "high" : cons > 0.22 ? "moderate" : "low"}</b> (${cons.toFixed(3)}).` : "";
  $("#summary").innerHTML = `${votes} of ${Object.keys(d.model_probs).length} models lean <b class="fake">fake</b>. ` +
    `Confidence ${(d.confidence * 100).toFixed(0)}%.` + consTxt + modeTxt;

  const mnames = Object.keys(d.model_probs);
  mk("#modelChart", { type: "bar", data: { labels: mnames.map(n => `${MODEL_NAMES[n] || n} (w=${(d.weights[n] || 0).toFixed(2)})`),
    datasets: [{ data: mnames.map(n => d.model_probs[n] * 100), backgroundColor: mnames.map(n => d.model_probs[n] >= d.threshold ? FAKE : REAL), borderRadius: 6 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 100, title: { display: true, text: "Fake probability %" } } } } });

  const g = Object.entries(d.modality_contrib).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  mk("#modalityChart", { type: "bar", data: { labels: g.map(x => x[0]), datasets: [{ data: g.map(x => x[1]), backgroundColor: g.map(x => x[1] > 0 ? FAKE : REAL), borderRadius: 6 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "contribution (log-odds)" } } } } });

  $("#signals").innerHTML = "<tr><th>Signal</th><th>Pushes toward</th></tr>" + d.top_signals.map(([k, v]) =>
    `<tr><td>${pretty(k)}</td><td class="${v > 0 ? "fake" : "real"}">${v > 0 ? "FAKE" : "REAL"} (${v > 0 ? "+" : ""}${v.toFixed(3)})</td></tr>`).join("");

  const pct = v => Math.min(100, Math.max(0, (v - 0.12) / 0.22 * 100));
  $("#frames").innerHTML = d.frames.map((f, i) => `<div class="frame"><img src="${f.img}"/><div class="m">Frame ${i + 1}
    ${f.consistency != null ? `<div>Caption match ${f.consistency.toFixed(3)}</div><div class="bar"><i style="width:${pct(f.consistency)}%;background:${ACC}"></i></div>` : ""}
    ${f.speech_consistency != null ? `<div>Speech match ${f.speech_consistency.toFixed(3)}</div><div class="bar"><i style="width:${pct(f.speech_consistency)}%;background:#9b6cff"></i></div>` : ""}
    ${f.speech_consistency === undefined && f.attention != null ? `<div>Attention ${(f.attention * 100).toFixed(0)}%</div>` : ""}</div></div>`).join("");
  $("#transcript").textContent = d.transcript ? `"${d.transcript}"` : (d.version === 2 ? "No clear speech detected (music / silence)." : "");
  $("#transcriptCard").hidden = d.version !== 2;
  if (d.audio_stats) table("#astats", d.audio_stats); else $("#astats").innerHTML = "";

  table("#vstats", d.video_stats); table("#tstats", d.text_stats); table("#cstats", d.consistency);
}

// ---------- model card
let MI = null;
function modelCard(mode) {
  const m = MI; if (!m || !m.test_metrics) return;
  const tm = m.version >= 2 ? m.test_metrics[mode] : m.test_metrics;
  const cmx = m.version >= 2 ? m.confusion_matrix_test[mode] : m.confusion_matrix_test;
  const rows = Object.entries(tm);
  $("#metricsTable").innerHTML = "<tr><th>Model</th><th>Acc</th><th>Macro-F1</th><th>AUC</th></tr>" + rows.map(([n, r]) =>
    `<tr class="${n === "ENSEMBLE" ? "best" : ""}"><td>${MODEL_NAMES[n] || n}</td><td>${(r.accuracy * 100).toFixed(1)}%</td><td>${(r.macro_f1 * 100).toFixed(1)}%</td><td>${r.auc.toFixed(3)}</td></tr>`).join("") +
    `<tr><td colspan="4" style="text-align:left;color:#8b97ab">Train ${m.n_train} · Val ${m.n_val} · Test ${m.n_test} videos</td></tr>`;
  const c = cmx;
  $("#cm").innerHTML = `<div class="h"></div><div class="h">Pred REAL</div><div class="h">Pred FAKE</div>
    <div class="h">True REAL</div><div class="ok">${c[0][0]}</div><div class="bad">${c[0][1]}</div>
    <div class="h">True FAKE</div><div class="bad">${c[1][0]}</div><div class="ok">${c[1][1]}</div>`;
  const g = Object.entries(m.feature_group_importance_pct);
  mk("#groupChart", { type: "doughnut", data: { labels: g.map(x => x[0]), datasets: [{ data: g.map(x => x[1]), backgroundColor: ["#6c8cff", "#9b6cff", "#2ed39a", "#ffb547", "#ff5470", "#5ad1ff", "#c8d26b", "#ff8fd1", "#8b97ab"] }] },
    options: { plugins: { legend: { position: "right" } } } });
}
fetch("/api/model-info").then(r => r.json()).then(m => {
  MI = m; if (m.version < 2) $("#modeSel").hidden = true; modelCard("full");
  if (m.version !== 3) document.querySelectorAll(".v1plus").forEach(e => e.hidden = true);
  if (m.version === 3) {
    document.querySelectorAll(".v2only").forEach(e => e.hidden = true);
    const b = document.querySelector(".brand small"); if (b) b.textContent = "Video + Text Fake News Detector · v1+ (video-only mode when caption is empty)";
  }
});
$("#modeSel").onchange = e => modelCard(e.target.value);
