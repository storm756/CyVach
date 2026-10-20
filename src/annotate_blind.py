"""Blind second-annotator tool (Phase 12): an independent person labels the 180 owner-checked test items
WITHOUT seeing any model, Claude or owner label, so chance-corrected agreement (Cohen's kappa) can be reported.

Local only: Flask on 127.0.0.1, no external requests. Shows sender + scrubbed text + the label guideline.
Keys: 1 ham, 2 scam, 3 promo, 4 caution, b borderline toggle, u undo, s skip.
Every answer is appended to annotation/second_annotator.csv immediately (resume-safe).

Run:  .venv/Scripts/python.exe src/annotate_blind.py      -> http://127.0.0.1:5058
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, ROOT  # noqa: E402

OUT = ROOT / "annotation" / "second_annotator.csv"
chk = pd.read_csv(ROOT / "annotation" / "check_results.csv").drop_duplicates("item_id", keep="last")
test = pd.read_csv(ROOT / "data" / "test.csv", keep_default_na=False)
SAMPLE = test[test["item_id"].isin(set(chk["item_id"]))].sample(frac=1, random_state=4242).reset_index(drop=True)
ITEMS = [{"item_id": r.item_id, "sender": r.sender, "text": r.text} for r in SAMPLE.itertuples()]
FIELDS = ["item_id", "label", "borderline", "seconds", "timestamp"]


def load_done() -> dict:
    if not OUT.exists():
        return {}
    return {r["item_id"]: r for r in pd.read_csv(OUT, keep_default_na=False).to_dict("records")}


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
    assert a["label"] in LABELS + ["skip"]
    new = not OUT.exists()
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"item_id": a["item_id"], "label": a["label"], "borderline": int(a["borderline"]),
                    "seconds": round(float(a["seconds"]), 1), "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
    return jsonify({"ok": True})


@app.post("/api/undo")
def undo():
    if OUT.exists():
        d = pd.read_csv(OUT, keep_default_na=False)
        if len(d):
            d.iloc[:-1].to_csv(OUT, index=False)
    return jsonify({"ok": True})


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMS labelling</title>
<style>
:root{--bg:#f6f7f9;--fg:#16181d;--muted:#5c6370;--card:#fff;--line:#d9dde3;--acc:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e8eaed;--muted:#9aa0a6;--card:#1c1f24;--line:#30353c;--acc:#7aa2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>div{height:100%;background:var(--acc)}
.meta{color:var(--muted);font-size:14px;margin:8px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.sender{font-family:ui-monospace,Consolas,monospace;color:var(--muted);font-size:14px}
.text{font-size:18px;margin:10px 0 18px;white-space:pre-wrap;word-break:break-word}
.btns{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
button{font:inherit;padding:12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.bl{margin-top:10px;color:var(--muted);font-size:14px}
.rules{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin-bottom:16px;font-size:14px}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:12px}.done{text-align:center;padding:40px}
</style></head><body><main>
<h1 style="font-size:20px;margin:0 0 12px">Label each SMS (your own judgement)</h1>
<div class="rules"><b>Rules</b> (apply in order; if several fit, the first one wins):<br>
1. <b>scam</b>: tries to cheat you, steal money/passwords, install an app/malware, or asks you to share/forward an OTP. Also OTP-style texts sent from a personal mobile number.<br>
2. <b>caution</b>: genuine but security-sensitive: OTP or verification code, login/new-device alert, password/PIN change, unusual-activity alert.<br>
3. <b>promo</b>: honest marketing, offers, discounts, sales.<br>
4. <b>ham</b>: personal chat and ordinary notifications (debit/credit alerts, delivery, tickets, reminders).<br>
Mark <b>borderline</b> (<kbd>b</kbd>) if you are genuinely unsure. Text like &lt;NAME&gt; or &lt;URL&gt; is a privacy mask: ignore it.<br>
Keys: <kbd>1</kbd> ham · <kbd>2</kbd> scam · <kbd>3</kbd> promo · <kbd>4</kbd> caution · <kbd>b</kbd> borderline · <kbd>u</kbd> undo · <kbd>s</kbd> skip</div>
<div class="bar"><div id="prog" style="width:0"></div></div><div class="meta" id="count"></div>
<div id="view"></div>
</main><script>
const NAMES={ham:'ham',spam:'scam',promo:'promo',caution:'caution'};
let S=null,queue=[],cur=null,border=false,t0=0,history=[];
async function load(){S=await (await fetch('/api/state')).json();queue=S.items.filter(i=>!(i.item_id in S.done));render();}
function render(){const n=Object.keys(S.done).length;document.getElementById('count').textContent=`${n} / ${S.items.length} done`;
 document.getElementById('prog').style.width=(100*n/S.items.length)+'%';const v=document.getElementById('view');cur=queue[0];border=false;t0=performance.now();
 if(!cur){v.innerHTML='<div class="card done"><h2>All done, thank you!</h2><p>You can close this tab.</p></div>';return;}
 v.innerHTML=`<div class="card"><div class="sender">${esc(cur.sender||'(no sender)')}</div><div class="text">${esc(cur.text)}</div>
 <div class="btns">${S.labels.map((l,i)=>`<button onclick="act('${l}')">${i+1} ${NAMES[l]}</button>`).join('')}</div>
 <div class="bl" id="bl">Borderline: off</div></div>`;}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
async function act(label){if(!cur)return;const row={item_id:cur.item_id,label,borderline:border?1:0,seconds:(performance.now()-t0)/1000};
 await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(row)});
 S.done[cur.item_id]=row;history.push(cur);queue.shift();render();}
async function undo(){const last=history.pop();if(!last)return;await fetch('/api/undo',{method:'POST'});delete S.done[last.item_id];queue.unshift(last);render();}
document.addEventListener('keydown',e=>{if('1234'.includes(e.key)&&e.key.length===1)act(S.labels[+e.key-1]);
 else if(e.key==='b'){border=!border;document.getElementById('bl').textContent=border?'Borderline: ON':'Borderline: off';}
 else if(e.key==='u')undo();else if(e.key==='s')act('skip');});
load();
</script></body></html>"""

if __name__ == "__main__":
    print(f"{len(ITEMS)} items. Results -> {OUT}")
    app.run(host="127.0.0.1", port=5058, debug=False)
