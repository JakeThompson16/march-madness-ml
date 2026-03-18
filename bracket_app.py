"""
bracket_app.py
==============
Run from your project root (same level as src/ and data/):

    python bracket_app.py

Opens http://localhost:5000 in your browser.

/api/predict?name_a=Duke&seed_a=1&name_b=Siena&seed_b=16
  → instantiates Team objects (which call generate_team_features internally)
    and calls Game.generate_probabilities() — your exact model pipeline.

All 64 R1 probabilities are pre-computed at startup and cached.
Subsequent round probabilities are fetched on demand as you advance teams.
"""

import json
import sys
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ── Import your classes ───────────────────────────────────────────────────────
# bracket_app.py lives at project root alongside src/ and data/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bracket.team import Team
from src.bracket.game import Game
from data.bracket_2026 import REGIONS_2026

SEASON = 2026

# ── Pre-build all Team objects once (triggers generate_team_features once) ───
print("[bracket_app] Loading team features…")
TEAMS: dict[str, Team] = {}
for region_name, seed_map in REGIONS_2026.items():
    for seed, name in seed_map.items():
        TEAMS[name] = Team(team_name=name, seed=seed, season=SEASON)
print(f"[bracket_app] {len(TEAMS)} teams loaded.")


def predict(name_a: str, seed_a: int, name_b: str, seed_b: int) -> tuple[float, float]:
    """
    Instantiate (or reuse) Team objects and call Game.generate_probabilities().
    Returns (prob_a, prob_b).
    """
    # If the team is in the original bracket, reuse the cached Team object.
    # For later-round matchups the teams are always original bracket teams.
    ta = TEAMS.get(name_a) or Team(team_name=name_a, seed=seed_a, season=SEASON)
    tb = TEAMS.get(name_b) or Team(team_name=name_b, seed=seed_b, season=SEASON)

    # Override seed in case it changed (it won't for tournament teams, but be safe)
    ta.seed = seed_a
    tb.seed = seed_b

    game = Game(team_a=ta, team_b=tb, tournament_round=1)
    game.generate_probabilities()
    return round(float(game.a_prob), 4), round(float(game.b_prob), 4)


# ─────────────────────────────────────────────────────────────────────────────
# INLINE HTML  —  single-file bracket GUI
#
# Layout (ESPN style):
#   TOP ROW:    East   [R64→R32→S16→E8] | FF+Champ center | [E8→S16→R32→R64] West
#   BOT ROW:    South  [R64→R32→S16→E8] | FF center       | [E8→S16→R32→R64] Midwest
#
# Left regions:  R64 far-left,  rounds advance RIGHT toward center
# Right regions: R64 far-right, rounds advance LEFT  toward center
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>2026 NCAA Tournament</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
:root{
  --bg:#0d0f14;--surface:#13161d;--card:#1a1e28;--card2:#1f2435;
  --border:#252a38;--border2:#2e3448;
  --text:#dde2f0;--muted:#5a6180;--muted2:#252a38;
  --accent:#3b82f6;
  --win:#10b981;--win-bg:rgba(16,185,129,0.09);
  --gold:#f59e0b;--gold-bg:rgba(245,158,11,0.08);
  --east:#818cf8;--south:#f472b6;--west:#fb923c;--midwest:#2dd4bf;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:12px;min-height:100vh;overflow-x:auto}

/* HEADER */
.hdr{position:sticky;top:0;z-index:200;background:var(--surface);border-bottom:1px solid var(--border);
  padding:10px 18px;display:flex;align-items:center;justify-content:space-between;gap:12px}
.hdr-title{font-family:'Oswald',sans-serif;font-size:19px;font-weight:600;letter-spacing:3px;text-transform:uppercase;white-space:nowrap}
.hdr-title span{color:var(--accent)}
.champ-banner{font-family:'Oswald',sans-serif;font-size:13px;font-weight:500;letter-spacing:2px;
  color:var(--gold);text-transform:uppercase;white-space:nowrap;display:none}
.champ-banner.on{display:block}
.reset-btn{font-family:'Oswald',sans-serif;font-size:11px;font-weight:500;letter-spacing:2px;
  text-transform:uppercase;background:transparent;color:var(--muted);border:1px solid var(--border2);
  padding:5px 14px;border-radius:2px;cursor:pointer;transition:all .15s;white-space:nowrap}
.reset-btn:hover{color:var(--text);border-color:var(--muted)}

/* LAYOUT */
.bracket-outer{padding:16px 10px 60px;min-width:1760px}
.half{display:flex;align-items:flex-start;width:100%}
.half-gap{height:16px}

/* REGION */
.region-wrap{display:flex;flex-direction:column;flex:1;min-width:0}
.region-header{font-family:'Oswald',sans-serif;font-size:10px;font-weight:600;letter-spacing:3px;
  text-transform:uppercase;text-align:center;padding:4px 0 3px;border-bottom:2px solid;margin-bottom:4px}
.rh-East   {color:var(--east);   border-color:var(--east)}
.rh-South  {color:var(--south);  border-color:var(--south)}
.rh-West   {color:var(--west);   border-color:var(--west)}
.rh-Midwest{color:var(--midwest);border-color:var(--midwest)}
.region-rounds{display:flex;flex-direction:row}

/* ROUND COLUMN */
.round-col{display:flex;flex-direction:column;width:148px;flex-shrink:0}
.round-hdr{font-family:'Oswald',sans-serif;font-size:8px;font-weight:400;letter-spacing:2px;
  text-transform:uppercase;color:var(--muted2);text-align:center;padding:2px 0 5px}
.games-col{display:flex;flex-direction:column;flex:1}

/* MATCHUP */
.matchup{display:flex;flex-direction:column;margin:2px 3px}
.slot{display:flex;align-items:center;gap:5px;padding:4px 6px;background:var(--card);
  border:1px solid var(--border);height:28px;position:relative;overflow:hidden;
  transition:background .1s,border-color .1s}
.slot:first-child{border-radius:2px 2px 0 0;border-bottom-color:var(--bg)}
.slot:last-child {border-radius:0 0 2px 2px}
.slot.click{cursor:pointer}
.slot.click:hover{background:var(--card2);border-color:var(--accent);z-index:1}
.slot.won{background:var(--win-bg);border-color:var(--win)!important}
.slot.won:first-child{border-bottom-color:var(--bg)}
.slot.lost{opacity:.35}
.slot.tbd{opacity:.22}
.tseed{font-family:'Oswald',sans-serif;font-size:10px;color:var(--muted);width:14px;text-align:center;flex-shrink:0;line-height:1}
.slot.won .tseed{color:var(--win)}
.tname{font-family:'Oswald',sans-serif;font-size:13px;font-weight:500;flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1;letter-spacing:.2px}
.tprob{font-family:'Oswald',sans-serif;font-size:11px;color:var(--muted);
  flex-shrink:0;min-width:32px;text-align:right;line-height:1}
.slot.won .tprob{color:var(--win)}
.pbar{position:absolute;bottom:0;left:0;height:2px;background:var(--accent);transition:width .4s;opacity:.4}
.slot.won .pbar{background:var(--win);opacity:.7}

/* CENTER */
.center-col{display:flex;flex-direction:column;align-items:center;width:210px;flex-shrink:0;padding:0 6px}
.ff-label{font-family:'Oswald',sans-serif;font-size:9px;font-weight:500;letter-spacing:3px;
  text-transform:uppercase;color:var(--gold);text-align:center;padding:16px 0 6px}
.sep{width:80%;height:1px;background:var(--border);margin:10px 0}
.champ-lbl{font-family:'Oswald',sans-serif;font-size:9px;letter-spacing:3px;
  text-transform:uppercase;color:var(--gold);opacity:.8;text-align:center;margin-bottom:6px}
.champ-winner-box{font-family:'Oswald',sans-serif;font-size:15px;font-weight:700;
  color:var(--gold);text-align:center;padding:10px 14px;border:1px solid var(--gold);
  border-radius:3px;background:var(--gold-bg);letter-spacing:1px;text-transform:uppercase;
  min-width:170px;line-height:1.2;margin-top:8px}
.trophy{font-size:22px;line-height:1;text-align:center;margin-top:4px}

/* SCORE INPUTS */
.score-section{display:flex;flex-direction:column;align-items:center;gap:5px;width:100%;margin-top:10px}
.score-lbl{font-family:'Oswald',sans-serif;font-size:9px;letter-spacing:2px;
  text-transform:uppercase;color:var(--muted);text-align:center}
.score-row{display:flex;align-items:center;gap:6px;justify-content:center}
.score-team{font-family:'Oswald',sans-serif;font-size:10px;font-weight:500;color:var(--muted);
  max-width:62px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right}
.score-team.b{text-align:left}
.score-input{font-family:'Oswald',sans-serif;font-size:16px;font-weight:600;color:var(--text);
  background:var(--card);border:1px solid var(--border2);border-radius:2px;
  width:50px;height:32px;text-align:center;outline:none;transition:border-color .15s;
  -moz-appearance:textfield}
.score-input::-webkit-inner-spin-button,.score-input::-webkit-outer-spin-button{-webkit-appearance:none}
.score-input:focus{border-color:var(--accent)}
.score-dash{font-family:'Oswald',sans-serif;font-size:14px;color:var(--muted2)}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-title">2026 NCAA <span>Tournament</span></div>
  <div class="champ-banner" id="champBanner"></div>
  <button class="reset-btn" onclick="resetBracket()">↺ Reset</button>
</div>
<div class="bracket-outer" id="bracketOuter"></div>

<script>
// ══════════════════════════════════════════════════════
// BRACKET DATA — injected from data/bracket_2026.py
// ══════════════════════════════════════════════════════
const REGIONS_2026 = __REGIONS_JSON__;

// R1 matchup pairs (0-indexed in seed-sorted array):
// 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
const R1 = [[0,15],[7,8],[4,11],[3,12],[5,10],[2,13],[6,9],[1,14]];
const RNAMES = ["R64","R32","S16","E8"];

// ══════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════
let S = {};
function mkG(a, b){ return {a, b, pA:null, pB:null, winner:null}; }

function initState(){
  S = { regions:{}, ff:[null,null], champ:null, scoreA:'', scoreB:'' };

  for (const [rn, seedMap] of Object.entries(REGIONS_2026)){
    const teams = Object.entries(seedMap)
      .map(([s,n]) => ({name:n, seed:parseInt(s)}))
      .sort((a,b) => a.seed - b.seed);

    const rounds = [ R1.map(([a,b]) => mkG(teams[a], teams[b])) ];
    for (let r=1; r<4; r++)
      rounds.push(Array.from({length: 8>>r}, () => mkG(null,null)));
    S.regions[rn] = rounds;
  }

  S.ff    = [mkG(null,null), mkG(null,null)];
  S.champ = mkG(null,null);

  // Kick off R64 probability fetches for all regions
  for (const rn of Object.keys(REGIONS_2026))
    for (let m=0; m<8; m++) fetchProb(rn, 0, m);
}

// ══════════════════════════════════════════════════════
// FETCH PROB  →  /api/predict (calls Team + Game)
// ══════════════════════════════════════════════════════
const CACHE = {};

async function fetchProb(region, round, mIdx, isFf=false, isChamp=false){
  const g = isChamp ? S.champ
           : isFf   ? S.ff[mIdx]
           :           S.regions[region]?.[round]?.[mIdx];
  if (!g || !g.a || !g.b) return;

  // Cache key uses both names (not just seeds) for accuracy
  const key = `${g.a.name}|${g.b.name}`;
  if (CACHE[key]){
    g.pA = CACHE[key][0];
    g.pB = CACHE[key][1];
    render(); return;
  }

  try {
    const url = `/api/predict?name_a=${encodeURIComponent(g.a.name)}&seed_a=${g.a.seed}`+
                             `&name_b=${encodeURIComponent(g.b.name)}&seed_b=${g.b.seed}`;
    const res = await fetch(url);
    const d   = await res.json();
    CACHE[key] = [d.prob_a, d.prob_b];
    // Re-resolve game reference (state may have changed during async call)
    const g2 = isChamp ? S.champ
              : isFf   ? S.ff[mIdx]
              :           S.regions[region]?.[round]?.[mIdx];
    if (g2 && g2.a?.name===g.a.name && g2.b?.name===g.b.name){
      g2.pA = d.prob_a; g2.pB = d.prob_b; render();
    }
  } catch(e){ console.error('fetchProb failed', e); }
}

// ══════════════════════════════════════════════════════
// PICK WINNER
// ══════════════════════════════════════════════════════
function doPick(region, round, mIdx, slot, isFf, isChamp){
  const g = isChamp ? S.champ
           : isFf   ? S.ff[mIdx]
           :           S.regions[region][round][mIdx];
  if (!g || g.winner) return;
  const team = slot==='A' ? g.a : g.b;
  if (!team) return;
  g.winner = team;

  if (isChamp){ render(); return; }

  if (isFf){
    const cs = mIdx===0 ? 'a' : 'b';
    S.champ[cs]=team; S.champ.winner=null; S.champ.pA=null; S.champ.pB=null;
    fetchProb(null, null, null, false, true);
    render(); return;
  }

  if (round < 3){
    const nR=round+1, nM=Math.floor(mIdx/2), ns=mIdx%2===0?'a':'b';
    const ng = S.regions[region][nR][nM];
    ng[ns]=team; ng.winner=null; ng.pA=null; ng.pB=null;
    cascadeClear(region, nR, nM);
    fetchProb(region, nR, nM);
  } else {
    // E8 → Final Four
    // East→ff[0].a   West→ff[0].b   South→ff[1].a   Midwest→ff[1].b
    const ffIdx  = ['East','South'].includes(region) ? 0 : 1;
    const ffSlot = ['East','West'].includes(region)  ? 'a' : 'b';
    S.ff[ffIdx][ffSlot]=team;
    S.ff[ffIdx].winner=null; S.ff[ffIdx].pA=null; S.ff[ffIdx].pB=null;
    cascadeFfClear(ffIdx);
    fetchProb(null, null, ffIdx, true);
  }
  render();
}

function cascadeClear(region, fromR, fromM){
  const g = S.regions[region][fromR]?.[fromM];
  if (!g) return;
  g.winner = null;
  if (fromR < 3) cascadeClear(region, fromR+1, Math.floor(fromM/2));
  else {
    const ffIdx  = ['East','South'].includes(region) ? 0 : 1;
    const ffSlot = ['East','West'].includes(region)  ? 'a' : 'b';
    S.ff[ffIdx][ffSlot] = null;
    cascadeFfClear(ffIdx);
  }
}

function cascadeFfClear(ffIdx){
  S.ff[ffIdx].winner = null;
  const cs = ffIdx===0 ? 'a' : 'b';
  S.champ[cs]=null; S.champ.winner=null; S.champ.pA=null; S.champ.pB=null;
}

// ══════════════════════════════════════════════════════
// RENDER
// ══════════════════════════════════════════════════════
function slotHTML(team, prob, won, lost, canClick, onclk){
  if (!team)
    return `<div class="slot tbd"><span class="tseed">–</span>`+
           `<span class="tname" style="color:var(--muted)">TBD</span></div>`;
  const cls = ['slot', canClick?'click':'', won?'won':'', lost?'lost':''].filter(Boolean).join(' ');
  const pct = prob!==null ? Math.round(prob*100) : null;
  return `<div class="${cls}"${canClick?` onclick="${onclk}"`:''}>`+
    `<span class="tseed">${team.seed}</span>`+
    `<span class="tname">${team.name}</span>`+
    `<span class="tprob">${pct!==null ? pct+'%' : '…'}</span>`+
    `<div class="pbar" style="width:${pct!==null?pct:0}%"></div>`+
    `</div>`;
}

function gameHTML(g, onA, onB){
  const {a,b,pA,pB,winner}=g;
  const both=a&&b, canA=both&&!winner, canB=both&&!winner;
  const aWon=winner&&a&&winner.name===a.name;
  const bWon=winner&&b&&winner.name===b.name;
  return `<div class="matchup">`+
    slotHTML(a, pA, aWon, bWon&&!aWon, canA, onA)+
    slotHTML(b, pB, bWon, aWon&&!bWon, canB, onB)+
    `</div>`;
}

// One round column. Each column is divided into 8 equal flex zones;
// games are centered in their zone(s).
function roundColHTML(roundIdx, games, region, isFf, isChamp){
  const slotsPerGame = 8 / games.length; // 1, 2, 4, 8
  let rows = '';
  for (let m=0; m<games.length; m++){
    const g = games[m];
    const onA = isChamp ? `doPick(null,null,null,'A',false,true)`
               : isFf   ? `doPick(null,null,${m},'A',true,false)`
               :           `doPick('${region}',${roundIdx},${m},'A',false,false)`;
    const onB = isChamp ? `doPick(null,null,null,'B',false,true)`
               : isFf   ? `doPick(null,null,${m},'B',true,false)`
               :           `doPick('${region}',${roundIdx},${m},'B',false,false)`;
    rows += `<div style="flex:${slotsPerGame};display:flex;align-items:center">`+
            gameHTML(g, onA, onB)+`</div>`;
  }
  return `<div class="round-col">`+
    `<div class="round-hdr">${RNAMES[roundIdx]}</div>`+
    `<div class="games-col" style="flex:1;display:flex;flex-direction:column">${rows}</div>`+
    `</div>`;
}

// Emit round columns in the visual order we want left→right:
//   left  regions: [0,1,2,3]  → R64 far-left,  E8 closest to center
//   right regions: [3,2,1,0]  → E8 closest to center, R64 far-right
function regionHTML(rn, side){
  const rounds = S.regions[rn];
  const colorClass = 'rh-'+rn;
  const order = side==='left' ? [0,1,2,3] : [3,2,1,0];
  let cols = '';
  for (const r of order)
    cols += roundColHTML(r, rounds[r], rn, false, false);
  return `<div class="region-wrap">`+
    `<div class="region-header ${colorClass}">${rn}</div>`+
    `<div class="region-rounds">${cols}</div>`+
    `</div>`;
}

function champHTML(){
  const g = S.champ;
  let html = `<div class="champ-lbl">Championship</div>`;
  if (!g.a && !g.b){
    html += `<div style="font-family:'Oswald',sans-serif;font-size:11px;color:var(--muted);`+
            `text-align:center;padding:8px 0;letter-spacing:1px">Awaiting Final Four</div>`;
  } else {
    html += gameHTML(g,
      `doPick(null,null,null,'A',false,true)`,
      `doPick(null,null,null,'B',false,true)`);
    if (g.winner){
      html += `<div class="trophy">🏆</div>`+
              `<div class="champ-winner-box">${g.winner.name}</div>`;
      const nameA=g.a?.name||'', nameB=g.b?.name||'';
      html += `<div class="score-section">`+
        `<div class="score-lbl">Final Score</div>`+
        `<div class="score-row">`+
          `<div class="score-team">${nameA}</div>`+
          `<input class="score-input" type="number" min="0" max="200" id="scoreA" `+
            `value="${S.scoreA}" placeholder="–" oninput="S.scoreA=this.value">`+
          `<div class="score-dash">–</div>`+
          `<input class="score-input" type="number" min="0" max="200" id="scoreB" `+
            `value="${S.scoreB}" placeholder="–" oninput="S.scoreB=this.value">`+
          `<div class="score-team b">${nameB}</div>`+
        `</div></div>`;
    }
  }
  return html;
}

function render(){
  const elA=document.getElementById('scoreA'), elB=document.getElementById('scoreB');
  if (elA) S.scoreA=elA.value;
  if (elB) S.scoreB=elB.value;

  const outer = document.getElementById('bracketOuter');

  const top =
    `<div class="half">`+
      regionHTML('East','left')+
      `<div class="center-col">`+
        `<div class="ff-label">Final Four</div>`+
        gameHTML(S.ff[0],`doPick(null,null,0,'A',true,false)`,`doPick(null,null,0,'B',true,false)`)+
        `<div class="sep"></div>`+
        champHTML()+
      `</div>`+
      regionHTML('West','right')+
    `</div>`;

  const bot =
    `<div class="half">`+
      regionHTML('South','left')+
      `<div class="center-col">`+
        `<div class="ff-label">Final Four</div>`+
        gameHTML(S.ff[1],`doPick(null,null,1,'A',true,false)`,`doPick(null,null,1,'B',true,false)`)+
      `</div>`+
      regionHTML('Midwest','right')+
    `</div>`;

  outer.innerHTML = top + `<div class="half-gap"></div>` + bot;

  const banner = document.getElementById('champBanner');
  if (S.champ.winner){
    banner.textContent = `🏆 2026 Champion: ${S.champ.winner.name}`;
    banner.classList.add('on');
  } else {
    banner.textContent = ''; banner.classList.remove('on');
  }
}

function resetBracket(){ S.scoreA=''; S.scoreB=''; initState(); render(); }

// BOOT
initState();
render();
</script>
</body>
</html>"""


# ── Inject bracket data from data/bracket_2026.py into the HTML ──────────────
_regions_json = json.dumps(REGIONS_2026)
HTML = HTML.replace("__REGIONS_JSON__", _regions_json)


# ── HTTP Server ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access log

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/predict":
            qs = parse_qs(parsed.query)
            try:
                name_a = qs["name_a"][0]
                seed_a = int(qs["seed_a"][0])
                name_b = qs["name_b"][0]
                seed_b = int(qs["seed_b"][0])
            except (KeyError, ValueError, IndexError):
                self._json(400, {"error": "name_a, seed_a, name_b, seed_b required"})
                return
            try:
                pa, pb = predict(name_a, seed_a, name_b, seed_b)
                self._json(200, {"prob_a": pa, "prob_b": pb})
            except Exception as e:
                print(f"[predict error] {name_a} vs {name_b}: {e}")
                self._json(500, {"error": str(e)})

        elif parsed.path in ("/", "/index.html"):
            self._html(200, HTML)
        else:
            self._json(404, {"error": "not found"})

    def _html(self, code, body):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)


def run(port=5000):
    server = HTTPServer(("localhost", port), Handler)
    url = f"http://localhost:{port}"
    print(f"\n  2026 NCAA Bracket  →  {url}\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run(port)