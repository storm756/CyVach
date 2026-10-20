"""Owner accept/override check of Claude's test labels (Plan B, owner decision 2026-10-18).

Local only: Flask on 127.0.0.1, no external requests, no CDN. Shows sender + text + Claude's proposed
label for a seeded, label-stratified sample of the test set (default 50 per class = 200 items).
Keys: Enter/Space = accept, 1 ham, 2 spam, 3 promo, 4 caution (override), b borderline toggle,
u undo, s skip. Every answer is appended to annotation/check_results.csv immediately (resume-safe).

Run:  .venv/Scripts/python.exe src/check_app.py            -> http://127.0.0.1:5057
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=50)
ap.add_argument("--seed", type=int, default=2718)
ap.add_argument("--out", default="annotation/check_results.csv")
ap.add_argument("--port", type=int, default=5057)
ARGS = ap.parse_args()
OUT = ROOT / ARGS.out

gold = pd.read_csv(ROOT / "annotation" / "gold_candidates.csv", keep_default_na=False)
cg = pd.read_csv(ROOT / "annotation" / "claude_gold_pass1.csv")
df = gold.merge(cg[["item_id", "claude_label"]], on="item_id")
SAMPLE = (pd.concat([g.sample(n=min(ARGS.per_class, len(g)), random_state=ARGS.seed) for _, g in df.groupby("claude_label")])
          .sample(frac=1, random_state=ARGS.seed).reset_index(drop=True))
ITEMS = [{"item_id": r.item_id, "sender": r.sender, "text": r.text, "proposed": r.claude_label} for r in SAMPLE.itertuples()]
FIELDS = ["item_id", "proposed", "final_label", "action", "borderline", "seconds", "timestamp"]


def load_done() -> dict:
    if not OUT.exists():
        return {}
    d = pd.read_csv(OUT, keep_default_na=False)
    return {r["item_id"]: r for r in d.to_dict("records")}


app = Flask(__name__)


@app.get("/")
def index():
    return PAGE


@app.get("/api/state")
def state():
    return jsonify({"items": ITEMS, "done": load_done(), "labels": LABELS})


@app.post("/api/answer")
def answer():
    a = request.get_json()
    assert a["final_label"] in LABELS + ["skip"]
    new = not OUT.exists()
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"item_id": a["item_id"], "proposed": a["proposed"], "final_label": a["final_label"],
                    "action": a["action"], "borderline": int(a["borderline"]), "seconds": round(float(a["seconds"]), 1),
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
    return jsonify({"ok": True})


@app.post("/api/undo")
def undo():
    """Remove the last answered row (append-only file rewritten without its last line)."""
    if OUT.exists():
        d = pd.read_csv(OUT, keep_default_na=False)
        if len(d):
            d.iloc[:-1].to_csv(OUT, index=False)
    return jsonify({"ok": True})


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CyVach label check</title>
<style>
:root{--bg:#f6f7f9;--fg:#16181d;--muted:#5c6370;--card:#fff;--line:#d9dde3;--acc:#2f5fd0;
--ham:#2e7d4f;--spam:#b3261e;--promo:#7a5af5;--caution:#b26a00}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e8eaed;--muted:#9aa0a6;--card:#1c1f24;--line:#30353c;--acc:#7aa2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>div{height:100%;background:var(--acc)}
.meta{display:flex;justify-content:space-between;color:var(--muted);font-size:14px;margin:8px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.sender{font-family:ui-monospace,Consolas,monospace;color:var(--muted);font-size:14px}
.text{font-size:18px;margin:10px 0 18px;white-space:pre-wrap;word-break:break-word}
.prop{font-size:15px}.pill{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-weight:600}
.btns{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:16px}
button{font:inherit;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.accept{background:var(--acc);color:#fff;border-color:var(--acc)}
.bl{margin-top:10px;color:var(--muted);font-size:14px}.bl b{color:var(--caution)}
details{margin-top:20px;color:var(--muted);font-size:14px}kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:12px}
.done{text-align:center;padding:40px}
</style></head><body><main>
<h1 style="font-size:20px;margin:0 0 12px">Check Claude's proposed labels</h1>
<div class="bar"><div id="prog" style="width:0"></div></div>
<div class="meta"><span id="count"></span><span id="ovr"></span></div>
<div id="view"></div>
<details><summary>Keys and label rules (cheat sheet)</summary>
<p><kbd>Enter</kbd>/<kbd>Space</kbd> accept · <kbd>1</kbd> ham · <kbd>2</kbd> spam (scam) · <kbd>3</kbd> promo · <kbd>4</kbd> caution · <kbd>b</kbd> borderline · <kbd>u</kbd> undo · <kbd>s</kbd> skip</p>
<p><b>spam</b>: tries to defraud, steal credentials or money, install malware; OTP-sharing requests; OTP-style texts from personal numbers.<br>
<b>caution</b>: legitimate but security-sensitive: OTP or verification code, login/new-device alert, password/PIN change, unusual activity.<br>
<b>promo</b>: non-deceptive marketing and offers.<br><b>ham</b>: personal chat and non-sensitive notifications (debits, delivery, tickets, reminders).<br>
Tie-break: spam &gt; caution &gt; promo &gt; ham. Mark <b>borderline</b> if genuinely ambiguous.</p></details>
</main><script>
const COLORS={ham:'var(--ham)',spam:'var(--spam)',promo:'var(--promo)',caution:'var(--caution)'};
let S=null,queue=[],cur=null,border=false,t0=0,history=[];
async function load(){S=await (await fetch('/api/state')).json();queue=S.items.filter(i=>!(i.item_id in S.done));render();}
function stats(){const d=Object.values(S.done).filter(r=>r.final_label!=='skip');const o=d.filter(r=>r.action==='override').length;
 document.getElementById('count').textContent=`${Object.keys(S.done).length} / ${S.items.length} done`;
 document.getElementById('ovr').textContent=d.length?`overrides: ${o} (${(100*o/d.length).toFixed(1)}%)`:'';
 document.getElementById('prog').style.width=(100*Object.keys(S.done).length/S.items.length)+'%';}
function render(){stats();const v=document.getElementById('view');cur=queue[0];border=false;t0=performance.now();
 if(!cur){v.innerHTML='<div class="card done"><h2>All done — thank you!</h2><p>Results are saved in annotation/check_results.csv. You can close this tab.</p></div>';return;}
 v.innerHTML=`<div class="card"><div class="sender">${esc(cur.sender||'(no sender)')}</div><div class="text">${esc(cur.text)}</div>
 <div class="prop">Claude's label: <span class="pill" style="background:${COLORS[cur.proposed]}">${cur.proposed}</span></div>
 <div class="btns"><button class="accept" onclick="act(null)">Accept ⏎</button>${S.labels.map((l,i)=>`<button onclick="act('${l}')">${i+1} ${l}</button>`).join('')}</div>
 <div class="bl" id="bl">Borderline: off (press <kbd>b</kbd>)</div></div>`;}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
async function act(label){if(!cur)return;const final=label||cur.proposed;const action=label===null?'accept':(label==='skip'?'skip':(final===cur.proposed?'accept':'override'));
 const row={item_id:cur.item_id,proposed:cur.proposed,final_label:final,action,borderline:border?1:0,seconds:(performance.now()-t0)/1000};
 await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(row)});
 S.done[cur.item_id]=row;history.push(cur);queue.shift();render();}
async function undo(){const last=history.pop();if(!last)return;await fetch('/api/undo',{method:'POST'});delete S.done[last.item_id];queue.unshift(last);render();}
document.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();act(null);}
 else if('1234'.includes(e.key)&&e.key.length===1)act(S.labels[+e.key-1]);
 else if(e.key==='b'){border=!border;document.getElementById('bl').innerHTML=border?'Borderline: <b>ON</b>':'Borderline: off (press <kbd>b</kbd>)';}
 else if(e.key==='u')undo();else if(e.key==='s')act('skip');});
load();
</script></body></html>"""

if __name__ == "__main__":
    print(f"{len(ITEMS)} items to check ({ARGS.per_class} per proposed class). Results -> {OUT}")
    app.run(host="127.0.0.1", port=ARGS.port, debug=False)
