"""
visify.render.templates.training_template
──────────────────────────────────────────
Renders the full interactive training visualization:
  - Loss curve (live animated line chart)
  - Accuracy meter
  - Network diagram with activation heatmap
  - Weight magnitude bars
  - Prediction table
  - Epoch scrubber
"""
from __future__ import annotations
import json
import uuid
from typing import Any, Dict, List


def render_training_html(frames: List[Dict], metadata: Dict[str, Any]) -> str:
    uid       = uuid.uuid4().hex[:8]
    frames_js = json.dumps(frames)
    layers    = metadata.get("layers", [2, 4, 1])
    act_name  = metadata.get("activation", "sigmoid")
    total_ep  = metadata.get("epochs", len(frames))

    return f"""
<div id="vt-{uid}" style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;border-radius:16px;padding:22px;color:#e2e8f0;max-width:960px;">
<style>
#vt-{uid} {{ --bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:rgba(255,255,255,0.08);
  --text:#e2e8f0;--text2:#8b949e;--text3:#484f58;
  --purple:#a371f7;--blue:#58a6ff;--green:#3fb950;
  --yellow:#d29922;--red:#f85149;--teal:#39d353; }}
#vt-{uid} *{{box-sizing:border-box;margin:0;padding:0}}
#vt-{uid} .topbar{{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}}
#vt-{uid} .title{{font-size:15px;font-weight:600;color:var(--text)}}
#vt-{uid} .badges{{display:flex;gap:6px}}
#vt-{uid} .badge{{font-size:10px;font-weight:600;padding:3px 9px;border-radius:20px;border:1px solid;text-transform:uppercase;letter-spacing:.07em}}
#vt-{uid} .badge.purple{{background:rgba(163,113,247,.12);color:var(--purple);border-color:rgba(163,113,247,.3)}}
#vt-{uid} .badge.blue{{background:rgba(88,166,255,.1);color:var(--blue);border-color:rgba(88,166,255,.25)}}
#vt-{uid} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}}
#vt-{uid} .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px}}
#vt-{uid} .card{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px}}
#vt-{uid} .card-title{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:10px}}
#vt-{uid} .stat-num{{font-size:26px;font-weight:700}}
#vt-{uid} .stat-sub{{font-size:11px;color:var(--text2);margin-top:2px}}
#vt-{uid} canvas{{display:block;width:100%;height:auto}}
#vt-{uid} .scrubber-wrap{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:12px}}
#vt-{uid} .scrubber-row{{display:flex;align-items:center;gap:12px}}
#vt-{uid} .scrubber-label{{font-size:12px;color:var(--text2);min-width:80px}}
#vt-{uid} input[type=range]{{flex:1;accent-color:var(--purple);height:4px}}
#vt-{uid} .epoch-badge{{font-size:12px;font-weight:600;color:var(--purple);min-width:80px;text-align:right;font-family:monospace}}
#vt-{uid} .controls{{display:flex;gap:8px;margin-top:10px}}
#vt-{uid} .btn{{padding:6px 16px;font-size:12px;font-weight:500;border:1px solid var(--border);border-radius:7px;background:var(--bg3);color:var(--text2);cursor:pointer;transition:all .15s}}
#vt-{uid} .btn:hover{{background:var(--bg);color:var(--text);border-color:rgba(255,255,255,.18)}}
#vt-{uid} .btn.play{{background:rgba(163,113,247,.18);color:var(--purple);border-color:rgba(163,113,247,.4)}}
#vt-{uid} .btn.play.running{{background:rgba(248,81,73,.12);color:var(--red);border-color:rgba(248,81,73,.3)}}
#vt-{uid} .pred-table{{width:100%;font-size:12px;border-collapse:collapse}}
#vt-{uid} .pred-table th{{text-align:left;color:var(--text3);font-weight:500;font-size:10px;text-transform:uppercase;letter-spacing:.07em;padding:0 0 8px}}
#vt-{uid} .pred-table td{{padding:5px 0;border-top:1px solid var(--border);color:var(--text2)}}
#vt-{uid} .pred-table .correct{{color:var(--green)}}
#vt-{uid} .pred-table .wrong{{color:var(--red)}}
#vt-{uid} .weight-bars{{display:flex;flex-direction:column;gap:6px}}
#vt-{uid} .w-row{{display:flex;align-items:center;gap:8px}}
#vt-{uid} .w-label{{font-size:10px;color:var(--text3);min-width:60px}}
#vt-{uid} .w-track{{flex:1;height:6px;background:var(--bg3);border-radius:3px;overflow:hidden}}
#vt-{uid} .w-fill{{height:100%;border-radius:3px;transition:width .3s}}
#vt-{uid} .acc-bar-wrap{{margin-top:8px}}
#vt-{uid} .acc-track{{height:8px;background:var(--bg3);border-radius:4px;overflow:hidden;margin-top:6px}}
#vt-{uid} .acc-fill{{height:100%;border-radius:4px;background:linear-gradient(90deg,#238636,#3fb950);transition:width .4s}}
</style>

<div class="topbar">
  <div class="title">Training Visualizer</div>
  <div class="badges">
    <span class="badge purple">visify</span>
    <span class="badge blue">{act_name} · {len(layers)} layers</span>
  </div>
</div>

<!-- Epoch scrubber -->
<div class="scrubber-wrap">
  <div class="scrubber-row">
    <span class="scrubber-label">Epoch</span>
    <input type="range" id="scrub-{uid}" min="0" max="0" value="0" oninput="VT_{uid}.goto(+this.value)">
    <span class="epoch-badge" id="ep-badge-{uid}">—</span>
  </div>
  <div class="controls">
    <button class="btn play" id="play-btn-{uid}" onclick="VT_{uid}.togglePlay()">▶ Play</button>
    <button class="btn" onclick="VT_{uid}.goto(0)">↺ Start</button>
    <button class="btn" onclick="VT_{uid}.goto(VT_{uid}.maxIdx())">⏭ End</button>
  </div>
</div>

<!-- Row 1: loss chart + accuracy + epoch stats -->
<div class="grid" style="grid-template-columns:2fr 1fr">
  <div class="card">
    <div class="card-title">Loss curve</div>
    <canvas id="loss-cvs-{uid}" height="130"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Accuracy</div>
    <div class="stat-num" id="acc-num-{uid}" style="color:var(--green)">—</div>
    <div class="stat-sub" id="acc-sub-{uid}">of training samples</div>
    <div class="acc-bar-wrap">
      <div class="acc-track"><div class="acc-fill" id="acc-fill-{uid}" style="width:0%"></div></div>
    </div>
    <div style="margin-top:16px">
      <div class="card-title">Loss</div>
      <div class="stat-num" id="loss-num-{uid}" style="font-size:20px;color:var(--yellow)">—</div>
    </div>
  </div>
</div>

<!-- Row 2: network diagram -->
<div class="card" style="margin-bottom:12px">
  <div class="card-title">Network — avg activations this epoch</div>
  <canvas id="net-cvs-{uid}" height="180"></canvas>
</div>

<!-- Row 3: weights + predictions -->
<div class="grid">
  <div class="card">
    <div class="card-title">Weight magnitudes per layer</div>
    <div class="weight-bars" id="w-bars-{uid}"></div>
  </div>
  <div class="card">
    <div class="card-title">Sample predictions</div>
    <table class="pred-table">
      <thead><tr><th>Input</th><th>Target</th><th>Output</th><th></th></tr></thead>
      <tbody id="pred-body-{uid}"></tbody>
    </table>
  </div>
</div>

<script>
(function(){{
const uid = '{uid}';
const FRAMES = {frames_js};
const LAYERS = {json.dumps(layers)};
const DPR = window.devicePixelRatio || 1;
let cur = 0, playTimer = null;

function setupCanvas(id, h) {{
  const c = document.getElementById(id);
  const W = c.parentElement.clientWidth - 28;
  c.width  = W * DPR; c.height = h * DPR;
  c.style.width = W+'px'; c.style.height = h+'px';
  const ctx = c.getContext('2d'); ctx.scale(DPR, DPR);
  return [ctx, W, h];
}}

// ── Loss chart ──────────────────────────────────────────────────────────
function drawLoss(upTo) {{
  const [ctx, W, H] = setupCanvas('loss-cvs-'+uid, 130);
  if (!FRAMES.length) return;
  const PAD = {{t:10,r:10,b:28,l:46}};
  const cW = W - PAD.l - PAD.r, cH = H - PAD.t - PAD.b;

  const losses = FRAMES.slice(0, upTo+1).map(f=>f.loss);
  const maxL = Math.max(...losses, 0.01);
  const minL = Math.min(...losses, 0);

  // grid
  ctx.strokeStyle='rgba(255,255,255,0.05)'; ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){{
    const y=PAD.t + cH*(i/4);
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(PAD.l+cW,y); ctx.stroke();
    const val = maxL - (maxL-minL)*(i/4);
    ctx.fillStyle='rgba(139,148,158,0.6)'; ctx.font='9px sans-serif';
    ctx.textAlign='right'; ctx.fillText(val.toFixed(3), PAD.l-4, y+3);
  }}

  if(losses.length < 2) return;

  // gradient fill
  const grd = ctx.createLinearGradient(0, PAD.t, 0, PAD.t+cH);
  grd.addColorStop(0,'rgba(163,113,247,0.3)');
  grd.addColorStop(1,'rgba(163,113,247,0.0)');
  ctx.beginPath();
  losses.forEach((l,i)=>{{
    const x=PAD.l+cW*(i/(FRAMES.length-1||1));
    const y=PAD.t+cH*(1-(l-minL)/(maxL-minL||1));
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.lineTo(PAD.l+cW*(losses.length-1)/(FRAMES.length-1||1), PAD.t+cH);
  ctx.lineTo(PAD.l, PAD.t+cH); ctx.closePath();
  ctx.fillStyle=grd; ctx.fill();

  // line
  ctx.beginPath(); ctx.strokeStyle='#a371f7'; ctx.lineWidth=1.5;
  losses.forEach((l,i)=>{{
    const x=PAD.l+cW*(i/(FRAMES.length-1||1));
    const y=PAD.t+cH*(1-(l-minL)/(maxL-minL||1));
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.stroke();

  // dot at current
  const cx=PAD.l+cW*(upTo/(FRAMES.length-1||1));
  const cy=PAD.t+cH*(1-(losses[upTo]-minL)/(maxL-minL||1));
  ctx.beginPath(); ctx.arc(cx,cy,4,0,Math.PI*2);
  ctx.fillStyle='#a371f7'; ctx.fill();
  ctx.beginPath(); ctx.arc(cx,cy,4,0,Math.PI*2);
  ctx.strokeStyle='#0d1117'; ctx.lineWidth=1.5; ctx.stroke();

  // x-axis label
  ctx.fillStyle='rgba(139,148,158,0.5)'; ctx.font='9px sans-serif'; ctx.textAlign='center';
  ctx.fillText('epoch', PAD.l+cW/2, H-4);
}}

// ── Network diagram ──────────────────────────────────────────────────────
function drawNet(frame) {{
  const [ctx, W, H] = setupCanvas('net-cvs-'+uid, 180);
  const acts = frame.activations || LAYERS.map(n=>Array(n).fill(0));
  const PAD_X=50, PAD_Y=20;
  const usableW=W-PAD_X*2, usableH=H-PAD_Y*2;
  const colW = usableW/(LAYERS.length-1||1);

  const layout = LAYERS.map((n,li)=>{{
    const x = LAYERS.length===1 ? W/2 : PAD_X+li*colW;
    const r = Math.max(5, Math.min(14, 18-n));
    const gap = n>1 ? (usableH - n*r*2)/(n-1) : 0;
    return {{ x, r, neurons: Array.from({{length:n}},(_,ni)=>{{
      return {{x, y:PAD_Y+ni*(r*2+Math.max(3,gap)), r}};
    }})}};
  }});

  // connections
  for(let li=0;li<layout.length-1;li++){{
    const A=layout[li], B=layout[li+1];
    const aA=acts[li]||[], aB=acts[li+1]||[];
    for(let ai=0;ai<A.neurons.length;ai++){{
      for(let bi=0;bi<B.neurons.length;bi++){{
        const strength=(aA[ai]||0+aB[bi]||0)/2;
        if(strength<0.05) continue;
        const nA=A.neurons[ai], nB=B.neurons[bi];
        ctx.beginPath();
        ctx.moveTo(nA.x+nA.r, nA.y+nA.r);
        ctx.lineTo(nB.x-nB.r, nB.y+nB.r);
        ctx.strokeStyle=`rgba(88,166,255,${{(0.04+strength*0.25).toFixed(2)}})`;
        ctx.lineWidth=0.6; ctx.stroke();
      }}
    }}
  }}

  // neurons
  layout.forEach((col,li)=>{{
    const layerActs=acts[li]||[];
    col.neurons.forEach((n,ni)=>{{
      const act=Math.min(1,Math.max(0,layerActs[ni]||0));
      const cx=n.x, cy=n.y+n.r;

      // glow
      if(act>0.15){{
        const g=ctx.createRadialGradient(cx,cy,0,cx,cy,n.r*3);
        g.addColorStop(0,`rgba(63,185,80,${{(act*0.5).toFixed(2)}})`);
        g.addColorStop(1,'transparent');
        ctx.beginPath(); ctx.arc(cx,cy,n.r*3,0,Math.PI*2);
        ctx.fillStyle=g; ctx.fill();
      }}

      // body
      const r=Math.round(act*180), g=Math.round(80+act*105), b=Math.round(60+act*40);
      ctx.beginPath(); ctx.arc(cx,cy,n.r,0,Math.PI*2);
      ctx.fillStyle=`rgb(${{r}},${{g}},${{b}})`; ctx.fill();
      ctx.strokeStyle=`rgba(255,255,255,${{0.06+act*0.25}})`; ctx.lineWidth=0.8; ctx.stroke();

      // value label for small layers
      if(col.neurons.length<=6 && n.r>=8){{
        ctx.font=`${{Math.round(n.r*0.72)}}px monospace`;
        ctx.textAlign='center'; ctx.fillStyle=act>0.4?'#fff':'rgba(200,230,200,0.6)';
        ctx.fillText(act.toFixed(2), cx, cy+3);
      }}
    }});

    // layer label
    const lastN=col.neurons[col.neurons.length-1];
    const labelY=lastN ? lastN.y+lastN.r*2+14 : H-10;
    ctx.font='9px sans-serif'; ctx.textAlign='center';
    ctx.fillStyle='rgba(139,148,158,0.5)';
    const lname=li===0?'Input':li===LAYERS.length-1?'Output':`Hidden ${{li}}`;
    ctx.fillText(lname, col.x, labelY);
  }});
}}

// ── Weight bars ──────────────────────────────────────────────────────────
function updateWeights(frame) {{
  const container=document.getElementById('w-bars-'+uid);
  const mags=frame.weight_mags||[];
  const globalMax=Math.max(...mags.flat(), 0.01);
  container.innerHTML=mags.map((layer,i)=>{{
    const avg=layer.length?layer.reduce((a,b)=>a+b,0)/layer.length:0;
    const maxW=Math.max(...layer,0.01);
    const pct=Math.round((avg/globalMax)*100);
    const col=avg>0.5?'#a371f7':avg>0.25?'#58a6ff':'#3fb950';
    return `<div class="w-row">
      <span class="w-label">Layer ${{i}}→${{i+1}}</span>
      <div class="w-track"><div class="w-fill" style="width:${{pct}}%;background:${{col}}"></div></div>
      <span style="font-size:10px;color:var(--text3);min-width:36px;text-align:right">${{avg.toFixed(3)}}</span>
    </div>`;
  }}).join('');
}}

// ── Predictions table ────────────────────────────────────────────────────
function updatePreds(frame) {{
  const tbody=document.getElementById('pred-body-'+uid);
  const preds=frame.predictions||[];
  tbody.innerHTML=preds.map(p=>{{
    const out=p.output[0], tgt=p.target[0];
    const correct=Math.round(out)===Math.round(tgt);
    const cls=correct?'correct':'wrong';
    const icon=correct?'✓':'✗';
    return `<tr>
      <td>[${'{'}${{p.input.map(v=>v.toFixed(1)).join(', ')}}${'}'}]</td>
      <td>${{tgt.toFixed(0)}}</td>
      <td class="${{cls}}">${{out.toFixed(3)}}</td>
      <td class="${{cls}}">${{icon}}</td>
    </tr>`;
  }}).join('');
}}

// ── Main render ──────────────────────────────────────────────────────────
function render() {{
  if(!FRAMES.length) return;
  const f=FRAMES[cur];

  document.getElementById('ep-badge-'+uid).textContent=`Epoch ${{f.epoch}} / ${{FRAMES[FRAMES.length-1].epoch}}`;
  document.getElementById('acc-num-'+uid).textContent=Math.round(f.accuracy*100)+'%';
  document.getElementById('acc-sub-'+uid).textContent=`${{Math.round(f.accuracy*100)}}% correct`;
  document.getElementById('acc-fill-'+uid).style.width=Math.round(f.accuracy*100)+'%';
  document.getElementById('loss-num-'+uid).textContent=f.loss.toFixed(5);
  document.getElementById('scrub-'+uid).value=cur;

  drawLoss(cur);
  drawNet(f);
  updateWeights(f);
  updatePreds(f);
}}

// ── Public API ────────────────────────────────────────────────────────────
window['VT_'+uid] = {{
  goto(i) {{ cur=Math.max(0,Math.min(FRAMES.length-1,i)); render(); }},
  maxIdx() {{ return FRAMES.length-1; }},
  togglePlay() {{
    const btn=document.getElementById('play-btn-'+uid);
    if(playTimer){{
      clearInterval(playTimer); playTimer=null;
      btn.classList.remove('running'); btn.textContent='▶ Play';
    }} else {{
      btn.classList.add('running'); btn.textContent='⏹ Stop';
      playTimer=setInterval(()=>{{
        if(cur>=FRAMES.length-1){{
          clearInterval(playTimer);playTimer=null;
          btn.classList.remove('running');btn.textContent='▶ Play';return;
        }}
        cur++; render();
      }},80);
    }}
  }},
}};

// init scrubber max
document.getElementById('scrub-'+uid).max=FRAMES.length-1;
render();
}})();
</script>
"""
