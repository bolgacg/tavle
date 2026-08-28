"""Build docs/index.html, the desk page, from the marts.

The page is static on purpose: it is rebuilt by the pipeline after every
run, carries its own data, and works without a server or a login. The
freshness and test panels are read from the run ledger and dbt's
run_results.json, so a red panel means the pipeline said so, not the page."""
import datetime as dt
import html
import json
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
OUT = ROOT / "docs" / "index.html"
RUN_RESULTS = ROOT / "dbt" / "target" / "run_results.json"


def q(con, sql, params=None):
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def jsonable(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return v


def collect(db=DB):
    con = duckdb.connect(str(db), read_only=True)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    data = {}
    data["built_at"] = now.isoformat(timespec="minutes") + "Z"
    data["daily"] = q(con, """
        select area, day_dk, round(avg_eur,2) as avg_eur, round(low_eur,2) as low_eur, round(high_eur,2) as high_eur,
               negative_hours, is_complete, round(dk1_minus_dk2_eur,2) as spread, round(eurdkk,4) as eurdkk
        from desk_daily where day_dk >= current_date - interval 400 day order by day_dk, area""")
    data["recent"] = q(con, """
        select instrument, interval_start_utc, interval_minutes, round(value,2) as value
        from prices where unit = 'EUR/MWh' and interval_start_utc >= now()::timestamp - interval 10 day
        order by interval_start_utc""")
    data["freshness"] = q(con, """
        select instrument, max(interval_start_utc) as last_interval, max(_fetched_at) as last_fetch,
               count(*) as n_rows
        from prices group by 1 order by 1""")
    data["seam"] = q(con, """
        select area, min(hour_utc) as first_hour, max(hour_utc) as last_hour, count(*) as n_hours,
               count(*) filter (where source = 'DayAheadPrices') as quarter_hour_hours
        from power_hourly group by 1 order by 1""")
    data["wind"] = q(con, """
        select round(wind_share, 3) as wind_share, round(price_eur, 1) as price_eur, area
        from power_context
        where hour_utc >= now()::timestamp - interval 365 day and wind_share is not null
        order by hour_utc""")
    data["runs"] = q(con, """
        select run_id, task, attempt, status, started, finished
        from ops.runs where run_id = (select max(run_id) from ops.runs) order by started""") if \
        con.execute("select count(*) from information_schema.tables where table_schema='ops'").fetchone()[0] else []
    con.close()
    tests = []
    if RUN_RESULTS.exists():
        rr = json.loads(RUN_RESULTS.read_text())
        for r in rr.get("results", []):
            uid = r.get("unique_id", "")
            if uid.startswith("test."):
                tests.append({"name": uid.split(".")[2], "status": r.get("status"),
                              "failures": r.get("failures")})
    data["tests"] = tests
    return json.loads(json.dumps(data, default=jsonable))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tavle</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Barlow+Condensed:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f2ee;--ink:#1d1d1b;--mute:#6b6a66;--line:#d9d6cf;--card:#fbfaf7;--s1:#2a78d6;--s2:#eb6834;--ok:#2e7d4f;--bad:#b3261e;--warn:#9a6700}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15161a;--ink:#ecebe6;--mute:#a09e97;--line:#2f3138;--card:#1c1e24;--s1:#3987e5;--s2:#d95926;--ok:#5cbf83;--bad:#ef6a5f;--warn:#e0b14a}}
:root[data-theme="dark"]{--bg:#15161a;--ink:#ecebe6;--mute:#a09e97;--line:#2f3138;--card:#1c1e24;--s1:#3987e5;--s2:#d95926;--ok:#5cbf83;--bad:#ef6a5f;--warn:#e0b14a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 "DM Sans",system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font:700 56px/1 "Barlow Condensed",sans-serif;letter-spacing:.5px;margin:0 0 6px}
h2{font:600 26px/1.1 "Barlow Condensed",sans-serif;margin:44px 0 12px;text-transform:uppercase;letter-spacing:1px}
.sub{color:var(--mute);max-width:720px;margin:0 0 8px}
.meta{font:13px "JetBrains Mono",monospace;color:var(--mute)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card .k{font:13px "JetBrains Mono",monospace;color:var(--mute)}.card .v{font:700 30px/1.1 "Barlow Condensed",sans-serif;margin-top:4px}
.card .v small{font:14px "DM Sans",sans-serif;color:var(--mute);margin-left:6px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.chart{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;position:relative;overflow-x:auto}
svg{display:block;width:100%;height:auto}
.legend{display:flex;gap:18px;font-size:13px;color:var(--mute);margin:8px 4px 0}.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font:12px "JetBrains Mono",monospace;padding:6px 8px;border-radius:4px;display:none;white-space:nowrap}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}th{font-weight:500;color:var(--mute)}td.num,th.num{text-align:right;font-family:"JetBrains Mono",monospace;font-size:13px}
.wrap{overflow-x:auto}
.controls{display:flex;gap:8px;margin:0 0 10px;flex-wrap:wrap}.controls button{font:14px "DM Sans",sans-serif;padding:6px 12px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;cursor:pointer}.controls button[aria-pressed="true"]{border-color:var(--ink)}
footer{margin-top:60px;color:var(--mute);font-size:14px;border-top:1px solid var(--line);padding-top:16px}
a{color:inherit}
</style>
</head>
<body>
<main>
<h1>tavle</h1>
<p class="sub">A small data platform for a trading desk: Danish day-ahead power at native resolution across the 2025 dataset seam, ECB reference rates beside it, one tested schema, and a run ledger. Rebuilt by the pipeline; nothing on this page is typed by hand.</p>
<p class="meta">built __BUILT__ &middot; source <a href="https://github.com/bolgacg/tavle">github.com/bolgacg/tavle</a></p>

<h2>Freshness and tests</h2>
<div class="grid" id="fresh"></div>
<div class="grid" id="tests"></div>

<h2>Last ten days, native resolution</h2>
<div class="controls" id="area-ctl"></div>
<div class="chart" id="recent"><div class="tip"></div></div>

<h2>Daily average, last 400 days</h2>
<div class="chart" id="daily"><div class="tip"></div></div>

<h2>DK1 minus DK2, daily average spread</h2>
<div class="chart" id="spread"><div class="tip"></div></div>

<h2>Wind share against price, last 365 days</h2>
<div class="chart" id="wind"><div class="tip"></div></div>

<h2>Last run</h2>
<div class="wrap"><table id="runs"></table></div>

<h2>Daily table, last 30 days</h2>
<div class="wrap"><table id="table"></table></div>

<footer>Power: Energinet, Energi Data Service (Elspotprices to 30 Sep 2025, DayAheadPrices from 1 Oct 2025). FX: European Central Bank reference rates. Prices are day-ahead auction results, EUR per MWh; the DKK column uses the ECB rate of the same day. This is a demonstration platform, not a data product.</footer>
</main>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const fmt = n => n == null ? "" : Number(n).toLocaleString("en-GB",{maximumFractionDigits:2});
const S = {DK1:"var(--s1)", DK2:"var(--s2)"};

function card(k,v,cls,small){return `<div class="card"><div class="k">${k}</div><div class="v ${cls||""}">${v}${small?`<small>${small}</small>`:""}</div></div>`}

(function freshness(){
  const now = new Date(D.built_at);
  $("#fresh").innerHTML = D.freshness.map(f=>{
    const last = new Date(f.last_interval);
    const ageH = (last - now)/36e5;
    const ok = f.instrument.endsWith("_DA") ? ageH > 0 : ageH > -96;
    const label = f.instrument.endsWith("_DA") ? (ageH>0?`covers +${ageH.toFixed(0)} h ahead`:`${(-ageH).toFixed(0)} h behind`) : `last ${f.last_interval.slice(0,10)}`;
    return card(f.instrument, ok?"fresh":"stale", ok?"ok":"bad", label + " &middot; " + fmt(f.n_rows) + " rows");
  }).join("");
  const t = D.tests, fails = t.filter(x=>x.status!=="pass");
  $("#tests").innerHTML = card("dbt tests", t.length? `${t.length-fails.length}/${t.length} passing` : "no run yet", fails.length||!t.length?"bad":"ok",
    fails.length? fails.map(f=>f.name).join(", ") : "gaps, seam, DST, business days, uniqueness") +
    D.seam.map(s=>card(s.area+" hourly grid", fmt(s.n_hours), "", `hours ${s.first_hour.slice(0,10)} to ${s.last_hour.slice(0,10)}`)).join("");
})();

function line(el, series, opts){
  const W=1000,H=300,P={l:48,r:12,t:12,b:28};
  const xs = series.flatMap(s=>s.pts.map(p=>p.x)), ys = series.flatMap(s=>s.pts.map(p=>p.y));
  const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(0,...ys), y1=Math.max(...ys);
  const X=x=>P.l+(x-x0)/(x1-x0||1)*(W-P.l-P.r), Y=y=>P.t+(1-(y-y0)/(y1-y0||1))*(H-P.t-P.b);
  const ticks=5, yt=[...Array(ticks+1)].map((_,i)=>y0+(y1-y0)*i/ticks);
  let g = yt.map(v=>`<line x1="${P.l}" x2="${W-P.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" stroke-width="1"/><text x="${P.l-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--mute)">${fmt(v)}</text>`).join("");
  if (y0<0) g += `<line x1="${P.l}" x2="${W-P.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--mute)" stroke-width="1"/>`;
  const xt=[...Array(5)].map((_,i)=>x0+(x1-x0)*i/4);
  g += xt.map(v=>`<text x="${X(v)}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--mute)">${opts.xlabel(v)}</text>`).join("");
  const paths = series.map(s=>`<path d="${s.pts.map((p,i)=>(i?"L":"M")+X(p.x).toFixed(1)+" "+Y(p.y).toFixed(1)).join(" ")}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`).join("");
  el.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.label}">${g}${paths}<line id="xh" x1="0" x2="0" y1="${P.t}" y2="${H-P.b}" stroke="var(--mute)" stroke-dasharray="3 3" style="display:none"/></svg>
    <div class="legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>`);
  const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), xh = svg.querySelector("#xh");
  svg.addEventListener("mousemove", e=>{
    const r = svg.getBoundingClientRect(); const mx = (e.clientX-r.left)/r.width*W;
    const xv = x0 + (mx-P.l)/(W-P.l-P.r)*(x1-x0);
    const rows = series.map(s=>{let b=s.pts[0];for(const p of s.pts){if(Math.abs(p.x-xv)<Math.abs(b.x-xv))b=p}return `${s.name} ${fmt(b.y)}`});
    const near = series[0].pts.reduce((b,p)=>Math.abs(p.x-xv)<Math.abs(b.x-xv)?p:b, series[0].pts[0]);
    xh.setAttribute("x1",X(near.x)); xh.setAttribute("x2",X(near.x)); xh.style.display="";
    tip.style.display="block"; tip.style.left=(e.clientX-r.left+12)+"px"; tip.style.top=(e.clientY-r.top-10)+"px";
    tip.textContent = opts.xlabel(near.x, true)+"  "+rows.join("  ");
  });
  svg.addEventListener("mouseleave", ()=>{tip.style.display="none"; xh.style.display="none"});
}

(function recent(){
  const areas=["DK1","DK2"]; let sel = new Set(areas);
  const ctl=$("#area-ctl");
  ctl.innerHTML = areas.map(a=>`<button aria-pressed="true" data-a="${a}">${a}</button>`).join("");
  function draw(){
    const el=$("#recent"); el.querySelectorAll("svg,.legend").forEach(n=>n.remove());
    const series = areas.filter(a=>sel.has(a)).map(a=>({name:a+" day-ahead, EUR/MWh", color:S[a],
      pts:D.recent.filter(r=>r.instrument===a+"_DA").map(r=>({x:Date.parse(r.interval_start_utc+"Z"), y:r.value}))}));
    if(series.length) line(el, series, {label:"recent prices", xlabel:(v,full)=>new Date(v).toISOString().slice(full?0:5, full?16:10).replace("T"," ")});
  }
  ctl.addEventListener("click", e=>{const b=e.target.closest("button"); if(!b) return; const a=b.dataset.a;
    if(sel.has(a)&&sel.size>1){sel.delete(a);b.setAttribute("aria-pressed","false")}else{sel.add(a);b.setAttribute("aria-pressed","true")} draw();});
  draw();
})();

(function daily(){
  const series=["DK1","DK2"].map(a=>({name:a+" daily average, EUR/MWh", color:S[a],
    pts:D.daily.filter(r=>r.area===a).map(r=>({x:Date.parse(r.day_dk), y:r.avg_eur}))}));
  line($("#daily"), series, {label:"daily averages", xlabel:v=>new Date(v).toISOString().slice(0,10)});
  const sp = D.daily.filter(r=>r.area==="DK1"&&r.spread!=null).map(r=>({x:Date.parse(r.day_dk), y:r.spread}));
  line($("#spread"), [{name:"DK1 minus DK2, EUR/MWh", color:"var(--s1)", pts:sp}], {label:"spread", xlabel:v=>new Date(v).toISOString().slice(0,10)});
})();

(function wind(){
  const el=$("#wind"), pts=D.wind; if(!pts.length) return;
  const W=1000,H=340,P={l:48,r:12,t:12,b:34};
  const x1=Math.max(...pts.map(p=>p.wind_share)), y0=Math.min(0,...pts.map(p=>p.price_eur)), y1=Math.max(...pts.map(p=>p.price_eur));
  const X=v=>P.l+v/(x1||1)*(W-P.l-P.r), Y=v=>P.t+(1-(v-y0)/(y1-y0||1))*(H-P.t-P.b);
  let g="";
  for(let i=0;i<=5;i++){const v=y0+(y1-y0)*i/5; g+=`<line x1="${P.l}" x2="${W-P.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${P.l-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--mute)">${fmt(v)}</text>`}
  for(let i=0;i<=5;i++){const v=x1*i/5; g+=`<text x="${X(v)}" y="${H-14}" text-anchor="middle" font-size="11" fill="var(--mute)">${(v*100).toFixed(0)}%</text>`}
  g += `<text x="${W/2}" y="${H-1}" text-anchor="middle" font-size="11" fill="var(--mute)">wind as a share of consumption</text>`;
  if (y0<0) g += `<line x1="${P.l}" x2="${W-P.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--mute)"/>`;
  const dots = pts.map(p=>`<circle cx="${X(p.wind_share).toFixed(1)}" cy="${Y(p.price_eur).toFixed(1)}" r="2" fill="${S[p.area]}" fill-opacity="0.28"/>`).join("");
  // binned median, the line a desk would actually read
  const bins={}; pts.forEach(p=>{const b=Math.min(19,Math.floor(p.wind_share/x1*20)); (bins[b]=bins[b]||[]).push(p.price_eur)});
  const med = Object.keys(bins).map(Number).sort((a,b)=>a-b).map(b=>{const v=bins[b].sort((a,c)=>a-c); return {x:(b+0.5)/20*x1, y:v[Math.floor(v.length/2)]}});
  const medPath = `<path d="${med.map((p,i)=>(i?"L":"M")+X(p.x).toFixed(1)+" "+Y(p.y).toFixed(1)).join(" ")}" fill="none" stroke="var(--ink)" stroke-width="2"/>`;
  el.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="wind share against price">${g}${dots}${medPath}</svg>
    <div class="legend"><span><i style="background:var(--s1)"></i>DK1 hour</span><span><i style="background:var(--s2)"></i>DK2 hour</span><span><i style="background:var(--ink)"></i>median by wind share</span></div>`);
})();

(function tables(){
  $("#runs").innerHTML = D.runs.length ? `<tr><th>task</th><th>attempt</th><th>status</th><th>started</th><th>finished</th></tr>` +
    D.runs.map(r=>`<tr><td>${r.task}</td><td class="num">${r.attempt}</td><td class="${r.status==="ok"?"ok":"bad"}">${r.status}</td><td>${(r.started||"").slice(0,19)}</td><td>${(r.finished||"").slice(0,19)}</td></tr>`).join("")
    : `<tr><td>no run recorded yet</td></tr>`;
  const rows = D.daily.slice(-60);
  $("#table").innerHTML = `<tr><th>day</th><th>area</th><th class="num">avg</th><th class="num">low</th><th class="num">high</th><th class="num">neg. hours</th><th class="num">DK1 minus DK2</th><th class="num">EURDKK</th></tr>` +
    rows.map(r=>`<tr><td>${r.day_dk}${r.is_complete?"":" <span class=\"warn\">partial</span>"}</td><td>${r.area}</td><td class="num">${fmt(r.avg_eur)}</td><td class="num">${fmt(r.low_eur)}</td><td class="num">${fmt(r.high_eur)}</td><td class="num">${r.negative_hours}</td><td class="num">${r.area==="DK1"?fmt(r.spread):""}</td><td class="num">${r.eurdkk??""}</td></tr>`).join("");
})();
</script>
</body>
</html>
"""


def build(db=DB, out=OUT):
    data = collect(db)
    page = PAGE.replace("__DATA__", json.dumps(data)).replace("__BUILT__", html.escape(data["built_at"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out, data


if __name__ == "__main__":
    path, data = build()
    print(f"wrote {path}: {len(data['daily'])} daily rows, {len(data['recent'])} recent rows, {len(data['tests'])} tests")
