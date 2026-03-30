"""
visify.render.templates.nn_template
────────────────────────────────────
Generates the interactive HTML visualization for neural networks.
Designed to render correctly in VS Code notebook output cells.
"""
from __future__ import annotations
import json
import uuid
from typing import Any, Dict, List


def render_nn_html(frames: List[Dict], metadata: Dict[str, Any]) -> str:
    uid = uuid.uuid4().hex[:8]
    frames_json = json.dumps(frames)

    return f"""
<div id="visify-{uid}" style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
<style>
#visify-{uid} {{
  --bg: #0f1117;
  --bg2: #1a1d27;
  --bg3: #22263a;
  --border: rgba(255,255,255,0.08);
  --text: #e2e8f0;
  --text2: #94a3b8;
  --text3: #64748b;
  --accent: #6366f1;
  --accent2: #818cf8;
  --teal: #2dd4bf;
  --amber: #fbbf24;
  --green: #34d399;
  --red: #f87171;
  --radius: 10px;
  background: var(--bg);
  border-radius: 14px;
  padding: 20px;
  color: var(--text);
  user-select: none;
  max-width: 900px;
}}
#visify-{uid} .header {{
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}}
#visify-{uid} .badge {{
  font-size: 10px; font-weight: 600; letter-spacing: .08em;
  padding: 3px 9px; border-radius: 20px;
  background: rgba(99,102,241,0.15); color: var(--accent2);
  border: 1px solid rgba(99,102,241,0.3);
  text-transform: uppercase;
}}
#visify-{uid} .step-info {{
  display: flex; flex-direction: column; gap: 2px;
}}
#visify-{uid} .step-label {{
  font-size: 14px; font-weight: 600; color: var(--text);
}}
#visify-{uid} .step-sub {{
  font-size: 11px; color: var(--text3); font-family: 'Fira Code', 'Cascadia Code', monospace;
  min-height: 16px;
}}
#visify-{uid} .canvas-wrap {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 0;
  margin-bottom: 14px;
  overflow: hidden;
}}
#visify-{uid} canvas {{
  display: block; width: 100%; height: auto;
}}
#visify-{uid} .controls {{
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}}
#visify-{uid} .btn {{
  padding: 6px 16px; font-size: 12px; font-weight: 500;
  border: 1px solid var(--border); border-radius: 7px;
  background: var(--bg3); color: var(--text2);
  cursor: pointer; transition: all .15s;
}}
#visify-{uid} .btn:hover {{ background: var(--bg); color: var(--text); border-color: rgba(255,255,255,0.18); }}
#visify-{uid} .btn.primary {{
  background: rgba(99,102,241,0.2); color: var(--accent2);
  border-color: rgba(99,102,241,0.4);
}}
#visify-{uid} .btn.primary:hover {{ background: rgba(99,102,241,0.35); }}
#visify-{uid} .btn:disabled {{ opacity: 0.35; cursor: default; }}
#visify-{uid} .step-dots {{
  display: flex; align-items: center; gap: 5px; margin-left: auto;
  flex-wrap: wrap; max-width: 200px;
}}
#visify-{uid} .dot {{
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--bg3); border: 1px solid var(--border);
  cursor: pointer; transition: all .2s;
}}
#visify-{uid} .dot.active {{
  background: var(--accent); border-color: var(--accent2);
  transform: scale(1.3);
}}
#visify-{uid} .legend {{
  display: flex; gap: 14px; flex-wrap: wrap; margin-top: 12px;
  padding-top: 12px; border-top: 1px solid var(--border);
}}
#visify-{uid} .leg {{
  display: flex; align-items: center; gap: 5px;
  font-size: 10px; color: var(--text3);
}}
#visify-{uid} .leg-dot {{
  width: 9px; height: 9px; border-radius: 50%;
}}
#visify-{uid} .stats-row {{
  display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap;
}}
#visify-{uid} .stat-card {{
  flex: 1; min-width: 80px;
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 12px;
}}
#visify-{uid} .stat-label {{
  font-size: 10px; color: var(--text3); margin-bottom: 2px;
}}
#visify-{uid} .stat-val {{
  font-size: 16px; font-weight: 600; color: var(--text);
}}
#visify-{uid} .auto-btn.running {{
  background: rgba(251,191,36,0.15); color: var(--amber);
  border-color: rgba(251,191,36,0.4);
}}
</style>

<div class="header">
  <div class="step-info">
    <span class="step-label" id="lbl-{uid}">Network architecture</span>
    <span class="step-sub" id="sub-{uid}">Loading…</span>
  </div>
  <span class="badge">visify · neural net</span>
</div>

<div class="stats-row" id="stats-{uid}"></div>

<div class="canvas-wrap">
  <canvas id="cvs-{uid}"></canvas>
</div>

<div class="controls">
  <button class="btn" id="prev-{uid}" onclick="VZ_{uid}.step(-1)">← Prev</button>
  <button class="btn primary" id="next-{uid}" onclick="VZ_{uid}.step(1)">Next →</button>
  <button class="btn auto-btn" id="auto-{uid}" onclick="VZ_{uid}.toggleAuto()">▶ Auto</button>
  <button class="btn" onclick="VZ_{uid}.reset()">↺ Reset</button>
  <div class="step-dots" id="dots-{uid}"></div>
</div>

<div class="legend">
  <div class="leg"><div class="leg-dot" style="background:#6366f1"></div>Highlighted layer</div>
  <div class="leg"><div class="leg-dot" style="background:#34d399"></div>High activation</div>
  <div class="leg"><div class="leg-dot" style="background:#2dd4bf"></div>Medium activation</div>
  <div class="leg"><div class="leg-dot" style="background:#1e3a5f"></div>Low / inactive</div>
  <div class="leg"><div class="leg-dot" style="background:rgba(255,255,255,0.12)"></div>Weight connection</div>
</div>
</div>

<script>
(function() {{
  const uid = '{uid}';
  const FRAMES = {frames_json};
  const cvs = document.getElementById('cvs-' + uid);
  const ctx = cvs.getContext('2d');
  let cur = 0, timer = null;

  const DPR = window.devicePixelRatio || 1;
  const W = 820, H = 340;
  cvs.width = W * DPR;
  cvs.height = H * DPR;
  cvs.style.width = W + 'px';
  cvs.style.height = H + 'px';
  ctx.scale(DPR, DPR);

  // ── color helpers ────────────────────────────────────────────────────
  function actColor(v, isHighlight) {{
    if (v <= 0) return isHighlight ? '#1a2a4a' : '#141824';
    const clamped = Math.min(1, Math.max(0, v));
    if (isHighlight) {{
      // purple-to-indigo for highlighted layer
      const r = Math.round(80 + clamped * 100);
      const g = Math.round(60 + clamped * 80);
      const b = Math.round(200 + clamped * 55);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}
    if (clamped < 0.4) {{
      // dark teal for low
      const t = clamped / 0.4;
      return `rgb(${{Math.round(20 + t*25)}},${{Math.round(50 + t*100)}},${{Math.round(80 + t*70)}})`;
    }}
    // teal-to-green for high
    const t = (clamped - 0.4) / 0.6;
    return `rgb(${{Math.round(45 + t*30)}},${{Math.round(150 + t*60)}},${{Math.round(150 + t*10)}})`;
  }}

  function glowColor(v, isHighlight) {{
    const alpha = Math.min(1, v * 1.4) * (isHighlight ? 0.7 : 0.45);
    if (isHighlight) return `rgba(99,102,241,${{alpha.toFixed(2)}})`;
    return `rgba(45,212,191,${{alpha.toFixed(2)}})`;
  }}

  // ── layout ───────────────────────────────────────────────────────────
  function computeLayout(layers) {{
    const PADDING_X = 55, PADDING_Y = 30;
    const usableW = W - PADDING_X * 2;
    const usableH = H - PADDING_Y * 2;
    const colW = usableW / (layers.length - 1 || 1);
    return layers.map((n, li) => {{
      const x = layers.length === 1 ? W / 2 : PADDING_X + li * colW;
      const maxR = 13, minR = 5;
      const r = Math.max(minR, Math.min(maxR, 20 - n));
      const totalH = n * r * 2 + (n - 1) * Math.max(4, 28 - n);
      const startY = (H - totalH) / 2;
      const gap = n > 1 ? (usableH - n * r * 2) / (n - 1) : 0;
      const neurons = Array.from({{length: n}}, (_, ni) => ({{
        x,
        y: PADDING_Y + ni * (r * 2 + Math.max(4, gap)),
        r,
      }}));
      return {{ x, neurons, r, n }};
    }});
  }}

  // ── draw ─────────────────────────────────────────────────────────────
  function draw(frame) {{
    ctx.clearRect(0, 0, W, H);

    const layers = frame.layers;
    const activations = frame.activations;
    const hlLayer = frame.highlight_layer;
    const layout = computeLayout(layers);

    // connections
    if (layers.length > 1) {{
      for (let li = 0; li < layout.length - 1; li++) {{
        const layerA = layout[li], layerB = layout[li + 1];
        const isHlEdge = hlLayer === li || hlLayer === li + 1;
        const actsA = activations[li] || [];
        const actsB = activations[li + 1] || [];

        for (let ai = 0; ai < layerA.neurons.length; ai++) {{
          for (let bi = 0; bi < layerB.neurons.length; bi++) {{
            const nA = layerA.neurons[ai], nB = layerB.neurons[bi];
            const actA = actsA[ai] || 0, actB = actsB[bi] || 0;
            const strength = (actA + actB) / 2;

            if (!isHlEdge && strength < 0.05) continue;

            const alpha = isHlEdge
              ? 0.05 + strength * 0.3
              : 0.02 + strength * 0.12;

            ctx.beginPath();
            ctx.moveTo(nA.x + nA.r, nA.y + nA.r);
            ctx.lineTo(nB.x - nB.r, nB.y + nB.r);
            ctx.strokeStyle = isHlEdge
              ? `rgba(129,140,248,${{alpha.toFixed(2)}})`
              : `rgba(255,255,255,${{alpha.toFixed(2)}})`;
            ctx.lineWidth = isHlEdge ? 0.8 : 0.4;
            ctx.stroke();
          }}
        }}
      }}
    }}

    // neurons
    for (let li = 0; li < layout.length; li++) {{
      const col = layout[li];
      const isHL = hlLayer === li;
      const acts = activations[li] || [];

      // layer label
      const labelY = col.neurons.length > 0
        ? col.neurons[col.neurons.length - 1].y + col.r * 2 + 18
        : H - 20;
      ctx.font = '10px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = isHL ? 'rgba(129,140,248,0.9)' : 'rgba(148,163,184,0.5)';
      const lbl = li === 0 ? 'Input' : li === layers.length - 1 ? 'Output' : `Hidden ${{li}}`;
      ctx.fillText(lbl, col.x, labelY);
      ctx.font = '9px -apple-system, sans-serif';
      ctx.fillStyle = isHL ? 'rgba(99,102,241,0.7)' : 'rgba(100,116,139,0.4)';
      ctx.fillText(`${{col.n}} neurons`, col.x, labelY + 13);

      for (let ni = 0; ni < col.neurons.length; ni++) {{
        const n = col.neurons[ni];
        const act = acts[ni] || 0;
        const cx = n.x, cy = n.y + n.r;

        // glow
        if (act > 0.1 || isHL) {{
          const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, n.r * 3.5);
          grd.addColorStop(0, glowColor(act, isHL));
          grd.addColorStop(1, 'transparent');
          ctx.beginPath();
          ctx.arc(cx, cy, n.r * 3.5, 0, Math.PI * 2);
          ctx.fillStyle = grd;
          ctx.fill();
        }}

        // neuron body
        ctx.beginPath();
        ctx.arc(cx, cy, n.r, 0, Math.PI * 2);
        ctx.fillStyle = actColor(act, isHL);
        ctx.fill();

        // ring
        ctx.beginPath();
        ctx.arc(cx, cy, n.r, 0, Math.PI * 2);
        ctx.strokeStyle = isHL
          ? `rgba(99,102,241,${{0.5 + act * 0.5}})`
          : `rgba(255,255,255,${{0.05 + act * 0.2}})`;
        ctx.lineWidth = isHL ? 1.5 : 0.8;
        ctx.stroke();

        // activation value label for small layers
        if (col.n <= 6 && n.r >= 9 && act > 0) {{
          ctx.font = `${{Math.round(n.r * 0.75)}}px 'Fira Code', monospace`;
          ctx.textAlign = 'center';
          ctx.fillStyle = act > 0.5
            ? 'rgba(255,255,255,0.95)'
            : 'rgba(200,220,240,0.7)';
          ctx.fillText(act.toFixed(2), cx, cy + 4);
        }}
      }}
    }}
  }}

  // ── stats cards ───────────────────────────────────────────────────────
  function updateStats(frame) {{
    const container = document.getElementById('stats-' + uid);
    const layers = frame.layers;
    const params = layers.slice(1).reduce((acc, n, i) => acc + n * (layers[i] + 1), 0);
    const depth = layers.length;
    const maxNeurons = Math.max(...layers);
    const hlActs = frame.highlight_layer !== null && frame.highlight_layer !== undefined
      ? frame.activations[frame.highlight_layer] || []
      : [];
    const avgAct = hlActs.length
      ? (hlActs.reduce((a, b) => a + b, 0) / hlActs.length).toFixed(3)
      : '—';

    container.innerHTML = `
      <div class="stat-card">
        <div class="stat-label">Depth</div>
        <div class="stat-val">${{depth}}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Parameters</div>
        <div class="stat-val">${{params.toLocaleString()}}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Max width</div>
        <div class="stat-val">${{maxNeurons}}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Layer avg act.</div>
        <div class="stat-val" style="font-size:14px">${{avgAct}}</div>
      </div>
    `;
  }}

  // ── UI update ─────────────────────────────────────────────────────────
  function render() {{
    const f = FRAMES[cur];
    draw(f);
    updateStats(f);

    document.getElementById('lbl-' + uid).textContent = f.label || '';
    document.getElementById('sub-' + uid).textContent = f.subtitle || '';

    document.getElementById('prev-' + uid).disabled = cur === 0;
    document.getElementById('next-' + uid).disabled = cur === FRAMES.length - 1;

    const dots = document.getElementById('dots-' + uid);
    dots.innerHTML = FRAMES.map((_, i) =>
      `<div class="dot${{i === cur ? ' active' : ''}}" onclick="VZ_${{uid}}.goto(${{i}})"></div>`
    ).join('');
  }}

  // ── public API ────────────────────────────────────────────────────────
  window['VZ_' + uid] = {{
    step(d) {{
      cur = Math.max(0, Math.min(FRAMES.length - 1, cur + d));
      render();
    }},
    goto(i) {{
      cur = Math.max(0, Math.min(FRAMES.length - 1, i));
      render();
    }},
    reset() {{
      if (timer) {{ clearInterval(timer); timer = null; }}
      document.getElementById('auto-' + uid).classList.remove('running');
      document.getElementById('auto-' + uid).textContent = '▶ Auto';
      cur = 0; render();
    }},
    toggleAuto() {{
      const btn = document.getElementById('auto-' + uid);
      if (timer) {{
        clearInterval(timer); timer = null;
        btn.classList.remove('running');
        btn.textContent = '▶ Auto';
      }} else {{
        btn.classList.add('running');
        btn.textContent = '⏹ Stop';
        timer = setInterval(() => {{
          if (cur >= FRAMES.length - 1) {{
            clearInterval(timer); timer = null;
            btn.classList.remove('running');
            btn.textContent = '▶ Auto';
            return;
          }}
          cur++; render();
        }}, 1100);
      }}
    }},
  }};

  render();
}})();
</script>
"""
