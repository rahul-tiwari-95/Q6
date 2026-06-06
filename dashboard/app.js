// dashboard/app.js — shared helpers for index/run/replay pages.

const MODE_KEY = "q6_dashboard_mode";

export function applyMode() {
  const mode = localStorage.getItem(MODE_KEY) || "lab";
  document.body.classList.toggle("theater", mode === "theater");
  const btn = document.getElementById("mode-toggle");
  if (btn) btn.textContent = mode === "theater" ? "theater mode" : "lab mode";
}

export function bindModeToggle() {
  const btn = document.getElementById("mode-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const cur = localStorage.getItem(MODE_KEY) || "lab";
    localStorage.setItem(MODE_KEY, cur === "lab" ? "theater" : "lab");
    applyMode();
  });
}

export async function fetchJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return r.json();
}

export async function fetchJSONL(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  const text = await r.text();
  return text.split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
}

export function qs(name) {
  return new URLSearchParams(location.search).get(name);
}

export function fmtNumber(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (typeof n !== "number") return String(n);
  return n.toFixed(digits);
}

export function fmtDuration(seconds) {
  if (!seconds) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

// Tiny chart — pure canvas line plot. No external library required.
export function drawLineChart(canvas, series, opts = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const xs = series.map((_, i) => i);
  const ys = series.filter(v => !Number.isNaN(v));
  if (ys.length === 0) return;
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const padL = 40, padB = 22, padT = 8, padR = 8;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  // axes
  const muted = getComputedStyle(document.body).getPropertyValue("--muted") || "#888";
  const accent = opts.color || getComputedStyle(document.body).getPropertyValue("--accent") || "#7cd1ff";
  ctx.strokeStyle = muted; ctx.fillStyle = muted;
  ctx.font = "10px ui-monospace, monospace";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.fillText(yMax.toFixed(1), 2, padT + 8);
  ctx.fillText(yMin.toFixed(1), 2, padT + plotH);

  // optional zero line
  if (opts.zeroLine && yMin < 0 && yMax > 0) {
    const zy = padT + plotH - ((0 - yMin) / Math.max(1e-9, yMax - yMin)) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, zy); ctx.lineTo(padL + plotW, zy);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // line
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < series.length; i++) {
    const v = series[i];
    if (Number.isNaN(v)) continue;
    const x = padL + (i / Math.max(1, series.length - 1)) * plotW;
    const y = padT + plotH - ((v - yMin) / Math.max(1e-9, yMax - yMin)) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (opts.title) {
    ctx.fillStyle = muted;
    ctx.fillText(opts.title, padL + 4, padT + 12);
  }
}

// Cell colors (must match config.py constants).
export const CELL_COLORS = {
  0: "#1c1f26",  // wall
  1: "#fde68a",  // pellet
  2: "#60a5fa",  // krishna
  3: "#f87171",  // hunter
  4: "#a78bfa",  // greedy
  5: "#f59e0b",  // patroller
  6: "#0b0c0f",  // empty
};

export function renderGrid(canvas, grid, options = {}) {
  const ctx = canvas.getContext("2d");
  const cells = grid.length;
  const cellPx = Math.floor(canvas.width / cells);
  for (let r = 0; r < cells; r++) {
    for (let c = 0; c < cells; c++) {
      ctx.fillStyle = CELL_COLORS[grid[r][c]] || "#000";
      ctx.fillRect(c * cellPx, r * cellPx, cellPx, cellPx);
    }
  }
}
