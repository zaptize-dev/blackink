// Strata-style live monitoring dashboard served on GET /metrics (for browsers).
#pragma once

#include <string_view>

namespace ingine::serve {

inline constexpr std::string_view kMonitorHtml = R"HTML(<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BlackInk · Monitor</title>
<style>
  :root {
    --bg: #0c0f12;
    --card: #13171c;
    --card-border: #1e242c;
    --text: #e6edf3;
    --muted: #7d8896;
    --green: #32d583;
    --green-dim: rgba(50, 213, 131, 0.14);
    --blue: #53b1fd;
    --blue-dim: rgba(83, 177, 253, 0.14);
    --amber: #f79009;
    --amber-dim: rgba(247, 144, 9, 0.14);
    --red: #f04438;
    --red-dim: rgba(240, 68, 56, 0.14);
    --purple: #9e77ed;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.45;
    min-height: 100vh;
  }
  a { color: inherit; text-decoration: none; }

  /* Topbar */
  .topbar {
    height: 52px;
    border-bottom: 1px solid var(--card-border);
    padding: 0 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0e1216;
    position: sticky;
    top: 0;
    z-index: 20;
  }
  .topbar-left { display: flex; align-items: center; gap: 22px; }
  .brand {
    display: flex; align-items: center; gap: 9px;
    font-weight: 700; font-size: 15px; letter-spacing: -0.02em;
  }
  .brand-logo {
    width: 24px; height: 24px; border-radius: 6px;
    background: linear-gradient(135deg, #252b34, #14181e);
    border: 1px solid #2c3540;
    display: grid; place-items: center;
  }
  .nav { display: flex; align-items: center; gap: 6px; }
  .nav a {
    padding: 6px 12px; border-radius: 7px; color: var(--muted);
    font-weight: 500; font-size: 13px; display: flex; align-items: center; gap: 6px;
    transition: background .15s, color .15s;
  }
  .nav a:hover { color: var(--text); background: #181d24; }
  .nav a.active { color: var(--text); background: #1a2028; border: 1px solid #27303b; }
  .status-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 12px; border-radius: 999px;
    background: #151a20; border: 1px solid var(--card-border);
    font-family: var(--mono); font-size: 12px; color: var(--text);
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); display: inline-block; }
  .dot.green { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot.blue { background: var(--blue); box-shadow: 0 0 8px var(--blue); }
  .dot.amber { background: var(--amber); box-shadow: 0 0 8px var(--amber); }
  .dot.red { background: var(--red); box-shadow: 0 0 8px var(--red); }

  /* Layout */
  .wrap { max-width: 1520px; margin: 0 auto; padding: 18px 20px 32px; display: flex; flex-direction: column; gap: 14px; }
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px 18px;
  }

  /* Model State Bar */
  .state-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 10px; }
  .state-title-row { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  .state-title { font-weight: 600; font-size: 13.5px; color: var(--text); }
  .state-pills { display: flex; align-items: center; gap: 14px; font-size: 12px; color: var(--muted); }
  .state-pills span { display: flex; align-items: center; gap: 6px; transition: color .15s; }
  .state-pills span.on { color: var(--text); font-weight: 600; }
  .state-meta { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 7px; }
  .state-meta .right { font-family: var(--mono); color: var(--text); }
  .prog-track { height: 6px; border-radius: 999px; background: #1d232b; overflow: hidden; }
  .prog-fill {
    height: 100%; border-radius: 999px; width: 0%;
    background: linear-gradient(90deg, #17b26a, #32d583);
    transition: width .25s ease;
  }
  .prog-fill.prefill { background: linear-gradient(90deg, #2e90fa, #53b1fd); }
  .prog-fill.idle { width: 100%; opacity: 0.16; }

  /* KPI Grid */
  .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
  @media (max-width: 1120px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 600px) { .kpi-grid { grid-template-columns: 1fr; } }

  .kpi-head { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 12.5px; font-weight: 500; margin-bottom: 10px; }
  .kpi-body { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
  .kpi-val { font-size: 27px; font-weight: 700; letter-spacing: -0.025em; font-variant-numeric: tabular-nums; }
  .kpi-unit { font-size: 13px; font-weight: 500; color: var(--muted); margin-left: 3px; }
  .kpi-sub { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .kpi-right { text-align: right; }
  .kpi-right .val2 { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .spark { width: 100%; height: 32px; margin-top: 12px; display: block; }

  /* Bottom Grid */
  .bottom-grid { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 14px; }
  @media (max-width: 1020px) { .bottom-grid { grid-template-columns: 1fr; } }

  .section-title { font-size: 13.5px; font-weight: 600; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; }
  .donut-wrap { display: flex; justify-content: center; margin: 8px 0 18px; position: relative; }
  .donut-center {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; pointer-events: none;
  }
  .donut-pct { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .donut-sub { font-size: 11.5px; color: var(--muted); font-family: var(--mono); }

  .bar-row { margin-bottom: 12px; }
  .bar-row:last-child { margin-bottom: 0; }
  .bar-labels { display: flex; justify-content: space-between; font-size: 11.5px; margin-bottom: 5px; }
  .bar-labels .k { color: var(--muted); }
  .bar-labels .v { font-family: var(--mono); color: var(--text); }
  .bar-track { height: 5px; border-radius: 999px; background: #1d232b; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 999px; background: var(--green); transition: width .3s ease; }
  .bar-fill.blue { background: var(--blue); }
  .bar-fill.amber { background: var(--amber); }

  .mtp-bars { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; margin-top: 10px; }
  .mtp-col { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .mtp-col-track { width: 100%; height: 42px; background: #191f26; border-radius: 4px; display: flex; align-items: flex-end; overflow: hidden; }
  .mtp-col-fill { width: 100%; background: var(--green); border-radius: 4px 4px 0 0; transition: height .3s ease; }
  .mtp-col-lbl { font-size: 10px; color: var(--muted); font-family: var(--mono); }
  .mtp-col-val { font-size: 10px; color: var(--text); font-family: var(--mono); }

  /* Requests Table */
  .btn-pill {
    background: #1a2028; border: 1px solid #28313d; color: var(--text);
    padding: 4px 10px; border-radius: 6px; font-size: 11.5px; cursor: pointer;
  }
  .btn-pill:hover { background: #222933; }
  .tbl-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    text-align: right; color: var(--muted); font-weight: 600; font-size: 10.5px;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 8px 10px; border-bottom: 1px solid var(--card-border);
  }
  th:first-child, th:nth-child(2), td:first-child, td:nth-child(2) { text-align: left; }
  td {
    text-align: right; padding: 9px 10px; border-bottom: 1px solid #181d24;
    font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  tr:last-child td { border-bottom: none; }
  .badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 2px 8px; border-radius: 999px; font-family: var(--sans); font-size: 11.5px; font-weight: 500;
    background: #1b222a; border: 1px solid #28313c;
  }
  .badge.done { color: var(--green); background: var(--green-dim); border-color: rgba(50,213,131,.28); }
  .badge.running { color: var(--blue); background: var(--blue-dim); border-color: rgba(83,177,253,.28); }
  .badge.length { color: var(--amber); background: var(--amber-dim); border-color: rgba(247,144,9,.28); }
  .badge.error, .badge.cancelled { color: var(--red); background: var(--red-dim); border-color: rgba(240,68,56,.28); }
  .empty-row td { text-align: center !important; color: var(--muted); padding: 28px; font-family: var(--sans); }
</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-left">
    <a class="brand" href="/">
      <span class="brand-logo">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#32d583" stroke-width="2.2">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
        </svg>
      </span>
      <span>BlackInk</span>
    </a>
  </div>
  <div class="status-pill" id="top-pill">
    <span class="dot" id="top-dot"></span>
    <span id="top-status">Connecting…</span>
  </div>
</header>

<main class="wrap">
  <section class="card">
    <div class="state-header">
      <div class="state-title-row">
        <span class="state-title">Model state</span>
        <div class="state-pills" id="state-pills">
          <span data-s="idle"><i class="dot"></i>Idle</span>
          <span data-s="prefilling"><i class="dot"></i>Reading</span>
          <span data-s="generating"><i class="dot"></i>Generating</span>
          <span data-s="queued"><i class="dot"></i>Queued</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <span style="font-family:var(--mono);font-size:11.5px;padding:3px 9px;border-radius:999px;background:#181e26;border:1px solid #28323e;color:var(--text)" id="reasoning-tag">Thinking: —</span>
        <span style="font-family:var(--mono);font-size:11.5px;color:var(--muted)" id="model-tag">—</span>
      </div>
    </div>
    <div class="state-meta">
      <span id="state-action">Waiting for requests</span>
      <span class="right" id="state-right">0 tokens · 0.0 tok/s</span>
    </div>
    <div class="prog-track"><div class="prog-fill idle" id="state-prog"></div></div>
  </section>

  <section class="kpi-grid">
    <div class="card">
      <div class="kpi-head">Speed</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-decode">0.0</span><span class="kpi-unit">t/s</span></div>
          <div class="kpi-sub" id="k-decode-sub">Decode avg</div>
        </div>
        <div class="kpi-right">
          <div class="val2" id="k-prefill">0<span class="kpi-unit">t/s</span></div>
          <div class="kpi-sub" id="k-prefill-sub">Prefill last</div>
        </div>
      </div>
      <svg class="spark" id="sp-speed"></svg>
    </div>

    <div class="card">
      <div class="kpi-head">GPU load</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-gpu">0</span><span class="kpi-unit">%</span></div>
          <div class="kpi-sub" id="k-gpu-name">NVIDIA GPU</div>
        </div>
      </div>
      <svg class="spark" id="sp-gpu"></svg>
    </div>

    <div class="card">
      <div class="kpi-head">VRAM</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-vram">0.0</span><span class="kpi-unit" id="k-vram-tot">/ 16 GB</span></div>
          <div class="kpi-sub">NVFP4 + 104K Q4 KV + MTP</div>
        </div>
      </div>
      <svg class="spark" id="sp-vram"></svg>
    </div>

    <div class="card">
      <div class="kpi-head">GPU temp</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-temp">0</span><span class="kpi-unit">°C</span></div>
          <div class="kpi-sub" id="k-temp-sub">Thermal sensor</div>
        </div>
      </div>
      <svg class="spark" id="sp-temp"></svg>
    </div>

    <div class="card">
      <div class="kpi-head">Power</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-power">0</span><span class="kpi-unit">W</span></div>
          <div class="kpi-sub" id="k-power-lim">of 180 W limit</div>
        </div>
      </div>
      <svg class="spark" id="sp-power"></svg>
    </div>

    <div class="card">
      <div class="kpi-head">PCIe</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-pcie">—</span><span class="kpi-unit" id="k-pcie-w"></span></div>
          <div class="kpi-sub" id="k-pcie-thru">TX 0.0 / RX 0.0 MB/s</div>
        </div>
      </div>
      <svg class="spark" id="sp-pcie"></svg>
    </div>

    <div class="card">
      <div class="kpi-head">CPU</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-cpu">0</span><span class="kpi-unit">%</span></div>
          <div class="kpi-sub" id="k-cpu-sub">0 threads · RAM 0.0 / 0 GB</div>
        </div>
      </div>
      <svg class="spark" id="sp-cpu"></svg>
    </div>

    <div class="card">
      <div class="kpi-head">MTP Speculative</div>
      <div class="kpi-body">
        <div>
          <div><span class="kpi-val" id="k-mtp">0.00</span><span class="kpi-unit">tok/round</span></div>
          <div class="kpi-sub" id="k-mtp-sub">0 accepted / 0 drafts</div>
        </div>
      </div>
      <svg class="spark" id="sp-mtp"></svg>
    </div>
  </section>

  <section class="bottom-grid">
    <div class="card">
      <div class="section-title"><span>Context fill</span><span style="font-family:var(--mono);font-size:11.5px;color:var(--muted)" id="ctx-max">96k</span></div>
      <div class="donut-wrap">
        <svg width="142" height="142" viewBox="0 0 142 142">
          <circle cx="71" cy="71" r="56" fill="none" stroke="#1e252e" stroke-width="11" />
          <circle id="ctx-arc" cx="71" cy="71" r="56" fill="none" stroke="#32d583" stroke-width="11"
            stroke-linecap="round" stroke-dasharray="351.86" stroke-dashoffset="351.86"
            transform="rotate(-90 71 71)" style="transition:stroke-dashoffset .35s ease" />
        </svg>
        <div class="donut-center">
          <div class="donut-pct" id="ctx-pct">0%</div>
          <div class="donut-sub" id="ctx-ratio">0 / 98.3k</div>
        </div>
      </div>

      <div class="bar-row">
        <div class="bar-labels"><span class="k">VRAM Usage</span><span class="v" id="b-vram-txt">0.0 / 16.0 GiB</span></div>
        <div class="bar-track"><div class="bar-fill" id="b-vram" style="width:0%"></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-labels"><span class="k">System RAM</span><span class="v" id="b-ram-txt">0.0 / 0.0 GiB</span></div>
        <div class="bar-track"><div class="bar-fill blue" id="b-ram" style="width:0%"></div></div>
      </div>
      <div class="bar-row" style="margin-top:16px">
        <div class="bar-labels"><span class="k">MTP acceptance by draft position</span><span class="v" id="mtp-rate-lbl">0.0%</span></div>
        <div class="mtp-bars" id="mtp-bars"></div>
      </div>
    </div>

    <div class="card">
      <div class="section-title">
        <span>Recent requests</span>
        <button class="btn-pill" id="btn-show-all">Show all (0)</button>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Status</th>
              <th>Prompt</th>
              <th>Reused</th>
              <th>Output</th>
              <th>Tok/s</th>
              <th>MTP</th>
              <th>Duration</th>
              <th>TTFT</th>
            </tr>
          </thead>
          <tbody id="req-tbody">
            <tr class="empty-row"><td colspan="9">Nenhuma requisição registrada ainda</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</main>

<script>
const H_MAX = 60;
const hist = {
  speed: Array(H_MAX).fill(0),
  gpu: Array(H_MAX).fill(0),
  vram: Array(H_MAX).fill(0),
  temp: Array(H_MAX).fill(0),
  power: Array(H_MAX).fill(0),
  pcie: Array(H_MAX).fill(0),
  cpu: Array(H_MAX).fill(0),
  mtp: Array(H_MAX).fill(0),
};
let showAll = false;
document.getElementById('btn-show-all').onclick = () => { showAll = !showAll; renderLast(); };
let lastData = null;

function push(arr, v) { arr.push(v); if (arr.length > H_MAX) arr.shift(); }

function drawSpark(id, arr, color, maxFixed) {
  const svg = document.getElementById(id);
  const w = 260, h = 32;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  const max = maxFixed || Math.max(1, ...arr) * 1.15;
  const min = 0;
  const pts = arr.map((v, i) => {
    const x = (i / (arr.length - 1)) * w;
    const y = h - 2 - ((Math.min(max, Math.max(min, v)) - min) / (max - min)) * (h - 5);
    return [x.toFixed(1), y.toFixed(1)];
  });
  const line = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0] + ',' + p[1]).join(' ');
  const area = `${line} L${w},${h} L0,${h} Z`;
  svg.innerHTML = `<path d="${area}" fill="${color}" fill-opacity="0.13"></path>` +
                  `<path d="${line}" fill="none" stroke="${color}" stroke-width="1.7"></path>`;
}

function fmtNum(n) { return Math.round(n).toLocaleString('en-US'); }
function fmtK(n) { return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n); }
function fmtTime(ms) {
  const d = new Date(ms);
  return d.toTimeString().slice(0, 8);
}

function renderLast() {
  if (!lastData) return;
  const d = lastData;
  const m = d.model;
  const s = d.system;

  // Top status pill
  const topDot = document.getElementById('top-dot');
  const topTxt = document.getElementById('top-status');
  const pills = document.querySelectorAll('#state-pills span');
  pills.forEach(p => {
    const on = p.dataset.s === m.state;
    p.classList.toggle('on', on);
    const dot = p.querySelector('.dot');
    dot.className = 'dot' + (on ? (m.state === 'generating' ? ' green' : m.state === 'prefilling' ? ' blue' : m.state === 'queued' ? ' amber' : '') : '');
  });

  const prog = document.getElementById('state-prog');
  const act = document.getElementById('state-action');
  const right = document.getElementById('state-right');
  document.getElementById('model-tag').textContent = `${m.model_id} · max_ctx ${fmtK(m.max_model_len)}`;
  const rEffort = m.reasoning_effort || 'medium (0.7)';
  const rSrc = m.reasoning_source === 'harness' ? 'harness' : 'padrão';
  document.getElementById('reasoning-tag').textContent = `Thinking: ${rEffort} · ${rSrc}`;

  if (m.state === 'generating') {
    topDot.className = 'dot green';
    topTxt.textContent = `Generating: ${m.active_live_tps.toFixed(1)} tok/s`;
    act.textContent = 'Answering';
    right.textContent = `${fmtNum(m.active_output_tokens)} tokens · ${m.active_live_tps.toFixed(1)} tok/s`;
    prog.className = 'prog-fill';
    prog.style.width = Math.min(100, Math.max(8, (m.active_output_tokens / Math.max(128, m.active_output_tokens + 64)) * 100)) + '%';
  } else if (m.state === 'prefilling') {
    topDot.className = 'dot blue';
    topTxt.textContent = `Reading prompt (${fmtNum(m.active_prompt_tokens)} tok)`;
    act.textContent = 'Reading prompt';
    right.textContent = `${fmtNum(m.active_prompt_tokens)} prompt tokens`;
    prog.className = 'prog-fill prefill';
    prog.style.width = '45%';
  } else if (m.state === 'queued') {
    topDot.className = 'dot amber';
    topTxt.textContent = `Queued: ${m.requests_deferred}`;
    act.textContent = 'Queued';
    right.textContent = `${m.requests_deferred} waiting`;
    prog.className = 'prog-fill idle';
    prog.style.width = '100%';
  } else {
    topDot.className = 'dot';
    topTxt.textContent = `Idle · avg ${m.predicted_tokens_seconds.toFixed(1)} tok/s`;
    act.textContent = 'Idle';
    right.textContent = `${fmtNum(m.tokens_predicted_total)} total tokens · ${m.predicted_tokens_seconds.toFixed(1)} tok/s avg`;
    prog.className = 'prog-fill idle';
    prog.style.width = '100%';
  }

  // KPIs
  const liveSpeed = m.state === 'generating' && m.active_live_tps > 0 ? m.active_live_tps : m.predicted_tokens_seconds;
  document.getElementById('k-decode').textContent = liveSpeed.toFixed(1);
  document.getElementById('k-decode-sub').textContent = m.state === 'generating' ? 'Decode now' : 'Decode avg';
  const prefillVal = m.last_prefill_tps > 0 ? m.last_prefill_tps : m.prompt_tokens_seconds;
  document.getElementById('k-prefill').innerHTML = `${fmtNum(prefillVal)}<span class="kpi-unit">t/s</span>`;
  document.getElementById('k-prefill-sub').textContent = m.last_prefill_tps > 0 ? `Prefill last (avg ${fmtNum(m.prompt_tokens_seconds)})` : 'Prefill avg';

  document.getElementById('k-gpu').textContent = s.gpu_util_pct;
  document.getElementById('k-gpu-name').textContent = s.gpu_name;

  document.getElementById('k-vram').textContent = s.vram_used_gb.toFixed(1);
  document.getElementById('k-vram-tot').textContent = `/ ${Math.round(s.vram_total_gb)} GB`;

  document.getElementById('k-temp').textContent = s.gpu_temp_c;
  document.getElementById('k-power').textContent = Math.round(s.power_w);
  document.getElementById('k-power-lim').textContent = `of ${Math.round(s.power_limit_w)} W limit`;

  document.getElementById('k-pcie').textContent = s.pcie_gen ? `Gen${s.pcie_gen}` : '—';
  document.getElementById('k-pcie-w').textContent = s.pcie_width ? `x${s.pcie_width}` : '';
  document.getElementById('k-pcie-thru').textContent = `TX ${s.pcie_tx_mbs.toFixed(1)} / RX ${s.pcie_rx_mbs.toFixed(1)} MB/s`;

  document.getElementById('k-cpu').textContent = Math.round(s.cpu_util_pct);
  document.getElementById('k-cpu-sub').textContent = `${s.cpu_threads} threads · RAM ${s.ram_used_gb.toFixed(1)} / ${s.ram_total_gb.toFixed(0)} GB`;

  const accPerRound = m.spec_decode_num_drafts_total > 0 ? (m.spec_decode_num_accepted_tokens_total / m.spec_decode_num_drafts_total) : 0;
  document.getElementById('k-mtp').textContent = (1 + accPerRound).toFixed(2);
  document.getElementById('k-mtp-sub').textContent = `${fmtNum(m.spec_decode_num_accepted_tokens_total)} accepted / ${fmtNum(m.spec_decode_num_drafts_total)} rounds`;

  drawSpark('sp-speed', hist.speed, '#32d583');
  drawSpark('sp-gpu', hist.gpu, '#32d583', 100);
  drawSpark('sp-vram', hist.vram, '#32d583', Math.max(16, s.vram_total_gb));
  drawSpark('sp-temp', hist.temp, '#f79009', 95);
  drawSpark('sp-power', hist.power, '#32d583', Math.max(180, s.power_limit_w));
  drawSpark('sp-pcie', hist.pcie, '#53b1fd');
  drawSpark('sp-cpu', hist.cpu, '#53b1fd', 100);
  drawSpark('sp-mtp', hist.mtp, '#32d583', 8);

  // Context fill donut
  const ctxUsed = m.context_used || 0;
  const ctxMax = m.max_model_len || 106496;
  const ctxPct = Math.min(100, (ctxUsed / ctxMax) * 100);
  const circ = 2 * Math.PI * 56;
  document.getElementById('ctx-arc').style.strokeDashoffset = (circ * (1 - ctxPct / 100)).toFixed(2);
  document.getElementById('ctx-pct').textContent = ctxPct.toFixed(ctxPct > 0 && ctxPct < 1 ? 1 : 0) + '%';
  document.getElementById('ctx-ratio').textContent = `${fmtK(ctxUsed)} / ${fmtK(ctxMax)}`;
  document.getElementById('ctx-max').textContent = fmtK(ctxMax);

  const vramPct = s.vram_total_gb > 0 ? (s.vram_used_gb / s.vram_total_gb) * 100 : 0;
  document.getElementById('b-vram').style.width = vramPct.toFixed(1) + '%';
  document.getElementById('b-vram-txt').textContent = `${s.vram_used_gb.toFixed(1)} / ${s.vram_total_gb.toFixed(1)} GiB`;

  const ramPct = s.ram_total_gb > 0 ? (s.ram_used_gb / s.ram_total_gb) * 100 : 0;
  document.getElementById('b-ram').style.width = ramPct.toFixed(1) + '%';
  document.getElementById('b-ram-txt').textContent = `${s.ram_used_gb.toFixed(1)} / ${s.ram_total_gb.toFixed(1)} GiB`;

  // MTP per pos bars
  const posArr = m.spec_accepted_per_pos || [0,0,0,0,0,0,0];
  const drafts = Math.max(1, m.spec_decode_num_drafts_total || 0);
  document.getElementById('mtp-rate-lbl').textContent = `${(accPerRound * 100 / 7).toFixed(1)}% draft hit`;
  document.getElementById('mtp-bars').innerHTML = posArr.map((cnt, idx) => {
    const pct = Math.min(100, (cnt / drafts) * 100);
    return `<div class="mtp-col">
      <div class="mtp-col-val">${pct.toFixed(0)}%</div>
      <div class="mtp-col-track"><div class="mtp-col-fill" style="height:${pct.toFixed(1)}%"></div></div>
      <div class="mtp-col-lbl">P${idx}</div>
    </div>`;
  }).join('');

  // Recent requests table
  const reqs = m.recent_requests || [];
  document.getElementById('btn-show-all').textContent = showAll ? `Show top 10 (${reqs.length})` : `Show all (${reqs.length})`;
  const rows = showAll ? reqs : reqs.slice(0, 10);
  const tbody = document.getElementById('req-tbody');
  if (rows.length === 0) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="9">Nenhuma requisição registrada ainda</td></tr>';
  } else {
    tbody.innerHTML = rows.map(r => {
      const cls = r.status === 'running' ? 'running' : r.status === 'length' ? 'length' : (r.status === 'error' || r.status === 'cancelled') ? 'error' : 'done';
      const label = r.status.charAt(0).toUpperCase() + r.status.slice(1);
      const mtpStr = r.spec_rounds > 0 ? `+${(r.spec_accepted / r.spec_rounds).toFixed(2)}/r` : '—';
      return `<tr>
        <td>${fmtTime(r.timestamp_ms)}</td>
        <td><span class="badge ${cls}"><i class="dot ${cls === 'done' ? 'green' : cls === 'running' ? 'blue' : cls === 'length' ? 'amber' : 'red'}"></i>${label}</span></td>
        <td>${fmtNum(r.prompt_tokens)}</td>
        <td>${fmtNum(r.reused_tokens)}</td>
        <td>${fmtNum(r.output_tokens)}</td>
        <td style="color:var(--green);font-weight:600">${r.decode_tps.toFixed(1)}</td>
        <td>${mtpStr}</td>
        <td>${r.duration_s.toFixed(1)}s</td>
        <td>${Math.round(r.ttft_ms)}ms</td>
      </tr>`;
    }).join('');
  }
}

async function poll() {
  try {
    const r = await fetch('/api/metrics', { cache: 'no-store' });
    if (!r.ok) return;
    const d = await r.json();
    lastData = d;
    const m = d.model, s = d.system;
    const liveSpeed = m.state === 'generating' && m.active_live_tps > 0 ? m.active_live_tps : m.predicted_tokens_seconds;
    const accPerRound = m.spec_decode_num_drafts_total > 0 ? (1 + m.spec_decode_num_accepted_tokens_total / m.spec_decode_num_drafts_total) : 1;
    push(hist.speed, liveSpeed);
    push(hist.gpu, s.gpu_util_pct);
    push(hist.vram, s.vram_used_gb);
    push(hist.temp, s.gpu_temp_c);
    push(hist.power, s.power_w);
    push(hist.pcie, s.pcie_tx_mbs + s.pcie_rx_mbs);
    push(hist.cpu, s.cpu_util_pct);
    push(hist.mtp, accPerRound);
    renderLast();
  } catch (_) {}
}

poll();
setInterval(poll, 500);
</script>
</body>
</html>
)HTML";

}  // namespace ingine::serve
